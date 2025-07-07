import os
import faiss
import pickle
import logging
import shutil
import numpy as np
from datetime import date
from itertools import repeat
from sortedcontainers import SortedList
from .embedding import OpenAILongerThanContextEmb
from typing import List, Union, Dict, Any, Tuple, Callable
from .memory_functions import (
    ImportanceScoreInitialization,
    get_importance_score_initialization_func,
    R_ConstantInitialization,
    LinearCompoundScore,
    ExponentialDecay,
    LinearImportanceScoreChange,
)

class id_generator_func:
    def __init__(self):
        self.current_id = 0
    def __call__(self):
        self.current_id += 1
        return self.current_id - 1

class MemoryDB:
    def __init__(
        self,
        db_name: str,
        id_generator: Callable,
        jump_threshold_upper: float,
        jump_threshold_lower: float,
        logger: logging.Logger,
        emb_config: Dict[str, Any],
        importance_score_initialization: ImportanceScoreInitialization,
        recency_score_initialization: R_ConstantInitialization,
        compound_score_calculation: LinearCompoundScore,
        importance_score_change_access_counter: LinearImportanceScoreChange,
        decay_function: ExponentialDecay,
        clean_up_threshold_dict: Dict[str, float],
    ) -> None:
        self.db_name = db_name
        self.id_generator = id_generator
        self.jump_threshold_upper = jump_threshold_upper
        self.jump_threshold_lower = jump_threshold_lower
        self.emb_config = emb_config
        self.importance_score_initialization_func = importance_score_initialization
        self.recency_score_initialization_func = recency_score_initialization
        self.compound_score_calculation_func = compound_score_calculation
        self.decay_function = decay_function
        self.importance_score_change_access_counter = importance_score_change_access_counter
        self.clean_up_threshold_dict = dict(clean_up_threshold_dict)
        self.logger = logger
        
        self.emb_func = OpenAILongerThanContextEmb(**self.emb_config)
        self.emb_dim = self.emb_func.get_embedding_dimension()
        self.universe: Dict[str, Dict[str, Any]] = {}

    def add_new_symbol(self, symbol: str) -> None:
        cur_index = faiss.IndexFlatIP(self.emb_dim)
        cur_index = faiss.IndexIDMap2(cur_index)
        self.universe[symbol] = {
            "score_memory": SortedList(key=lambda x: x["important_score_recency_compound_score"]),
            "index": cur_index,
        }

    def add_memory(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        if symbol not in self.universe:
            self.add_new_symbol(symbol)
        if isinstance(text, str): text = [text]
        
        emb = self.emb_func(text)
        faiss.normalize_L2(emb)
        
        ids = [self.id_generator() for _ in range(len(text))]
        importance_scores = [self.importance_score_initialization_func() for _ in range(len(text))]
        recency_scores = [self.recency_score_initialization_func() for _ in range(len(text))]
        
        partial_scores = [
            self.compound_score_calculation_func.recency_and_importance_score(r, i)
            for r, i in zip(recency_scores, importance_scores)
        ]
        
        self.universe[symbol]["index"].add_with_ids(emb, np.array(ids))
        
        for i in range(len(text)):
            memory_data = {
                "text": text[i], "id": ids[i], "important_score": importance_scores[i],
                "recency_score": recency_scores[i], "delta": 0,
                "important_score_recency_compound_score": partial_scores[i],
                "access_counter": 0, "date": date,
            }
            self.universe[symbol]["score_memory"].add(memory_data)
            self.logger.info(f"[{self.db_name}] Added: {memory_data}")

    def query(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        if (symbol not in self.universe) or (len(self.universe[symbol]["score_memory"]) == 0) or (top_k == 0):
            return [], []
        
        max_len = len(self.universe[symbol]["score_memory"])
        top_k = min(top_k, max_len)
        
        cur_index = self.universe[symbol]["index"]
        emb = self.emb_func(query_text)
        faiss.normalize_L2(emb)
        
        temp_text_list, temp_score, temp_ids = [], [], []
        
        p1_dists, p1_ids = cur_index.search(emb, top_k)
        p1_dists, p1_ids = p1_dists[0].tolist(), p1_ids[0].tolist()
        
        for cur_sim, cur_id in zip(p1_dists, p1_ids):
            if cur_id == -1: continue
            cur_record = next((r for r in self.universe[symbol]["score_memory"] if r["id"] == cur_id), None)
            if cur_record:
                temp_text_list.append(cur_record["text"])
                temp_ids.append(cur_record["id"])
                temp_score.append(self.compound_score_calculation_func.merge_score(cur_sim, cur_record["important_score_recency_compound_score"]))

        p2_ids = [self.universe[symbol]["score_memory"][-i-1]["id"] for i in range(top_k)]
        if p2_ids:
            reconstructable_ids = []
            reconstructable_embs = []
            for i in p2_ids:
                reconstructed_emb = cur_index.reconstruct(i)
                if reconstructed_emb is not None:
                    reconstructable_ids.append(i)
                    reconstructable_embs.append(reconstructed_emb)

            if reconstructable_ids:
                p2_emb = np.vstack(reconstructable_embs)
                temp_index = faiss.IndexFlatIP(self.emb_dim)
                temp_index = faiss.IndexIDMap2(temp_index)
                temp_index.add_with_ids(p2_emb, np.array(reconstructable_ids))
                
                p2_dist, p2_ids_res = temp_index.search(emb, top_k)
                p2_dist, p2_ids_res = p2_dist[0].tolist(), p2_ids_res[0].tolist()
                
                for cur_sim, cur_id in zip(p2_dist, p2_ids_res):
                    if cur_id == -1: continue
                    cur_record = next((r for r in self.universe[symbol]["score_memory"] if r["id"] == cur_id), None)
                    if cur_record:
                        temp_text_list.append(cur_record["text"])
                        temp_ids.append(cur_record["id"])
                        temp_score.append(self.compound_score_calculation_func.merge_score(cur_sim, cur_record["important_score_recency_compound_score"]))

        score_rank = np.argsort(temp_score)[::-1]
        
        ret_text_list, ret_ids, seen_ids = [], [], set()
        for i in score_rank:
            if temp_ids[i] not in seen_ids:
                ret_text_list.append(temp_text_list[i])
                ret_ids.append(temp_ids[i])
                seen_ids.add(temp_ids[i])
            if len(ret_ids) >= top_k:
                break
        return ret_text_list, ret_ids

    def update_access_count_with_feed_back(self, symbol: str, ids: List[int], feedback: int) -> List[int]:
        if symbol not in self.universe: return []
        success_ids = []
        for cur_id in ids:
            for cur_record in self.universe[symbol]["score_memory"]:
                if cur_record["id"] == cur_id:
                    cur_record["access_counter"] += feedback
                    cur_record["important_score"] = self.importance_score_change_access_counter(cur_record["access_counter"], cur_record["important_score"])
                    cur_record["important_score_recency_compound_score"] = self.compound_score_calculation_func.recency_and_importance_score(cur_record["recency_score"], cur_record["important_score"])
                    success_ids.append(cur_id)
                    break
        return success_ids

    def _decay(self) -> None:
        for symbol in self.universe:
            for record in self.universe[symbol]["score_memory"]:
                record["recency_score"], record["important_score"], record["delta"] = self.decay_function(record["important_score"], record["delta"])
                record["important_score_recency_compound_score"] = self.compound_score_calculation_func.recency_and_importance_score(record["recency_score"], record["important_score"])

    def _clean_up(self) -> List[int]:
        ret_removed_ids = []
        for symbol in self.universe:
            remove_ids = [r["id"] for r in self.universe[symbol]["score_memory"] if (r["recency_score"] < self.clean_up_threshold_dict["recency_threshold"]) or (r["important_score"] < self.clean_up_threshold_dict["importance_threshold"])]
            if remove_ids:
                self.universe[symbol]["score_memory"] = SortedList([r for r in self.universe[symbol]["score_memory"] if r["id"] not in remove_ids], key=lambda x: x["important_score_recency_compound_score"])
                self.universe[symbol]["index"].remove_ids(np.array(remove_ids, dtype=np.int64))
                ret_removed_ids.extend(remove_ids)
        return ret_removed_ids

    def step(self) -> List[int]:
        self._decay()
        return self._clean_up()

    def prepare_jump(self) -> Tuple[Dict, Dict, List[int]]:
        jump_dict_up, jump_dict_down, id_to_remove = {}, {}, []
        for symbol in self.universe:
            to_jump_up = [r for r in self.universe[symbol]["score_memory"] if r["important_score"] >= self.jump_threshold_upper]
            to_jump_down = [r for r in self.universe[symbol]["score_memory"] if r["important_score"] < self.jump_threshold_lower]
            
            if to_jump_up:
                emb_list = [self.universe[symbol]["index"].reconstruct(r["id"]) for r in to_jump_up]
                valid_embs = [emb for emb in emb_list if emb is not None]
                if valid_embs:
                    jump_dict_up[symbol] = {"jump_object_list": to_jump_up, "emb_list": np.vstack(valid_embs)}
            
            if to_jump_down:
                emb_list = [self.universe[symbol]["index"].reconstruct(r["id"]) for r in to_jump_down]
                valid_embs = [emb for emb in emb_list if emb is not None]
                if valid_embs:
                    jump_dict_down[symbol] = {"jump_object_list": to_jump_down, "emb_list": np.vstack(valid_embs)}
            
            ids_to_delete = [r["id"] for r in to_jump_up] + [r["id"] for r in to_jump_down]
            if ids_to_delete:
                id_to_remove.extend(ids_to_delete)
                self.universe[symbol]["index"].remove_ids(np.array(ids_to_delete, dtype=np.int64))
                self.universe[symbol]["score_memory"] = SortedList(
                    [r for r in self.universe[symbol]["score_memory"] if r["id"] not in ids_to_delete],
                    key=lambda x: x["important_score_recency_compound_score"]
                )
        return jump_dict_up, jump_dict_down, id_to_remove

    def accept_jump(self, jump_dict: Tuple[Dict, Dict], direction: str) -> None:
        if direction not in ["up", "down"]: raise ValueError("direction must be 'up' or 'down'")
        target_dict = jump_dict[0] if direction == "up" else jump_dict[1]
        for symbol, data in target_dict.items():
            if symbol not in self.universe: self.add_new_symbol(symbol)
            new_ids = []
            for obj in data["jump_object_list"]:
                new_ids.append(obj["id"])
                if direction == "up":
                    obj["recency_score"] = self.recency_score_initialization_func()
                    obj["delta"] = 0
            self.universe[symbol]["score_memory"].update(data["jump_object_list"])
            self.universe[symbol]["index"].add_with_ids(data["emb_list"], np.array(new_ids, dtype=np.int64))

    def save_checkpoint(self, name: str, path: str, force: bool = False) -> None:
        save_path = os.path.join(path, name)
        if os.path.exists(save_path):
            if not force: raise FileExistsError(f"Path {save_path} exists")
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        
        state_dict_to_save = {
            "db_name": self.db_name,
            "id_generator": self.id_generator,
            "jump_threshold_upper": self.jump_threshold_upper,
            "jump_threshold_lower": self.jump_threshold_lower,
            "emb_config": self.emb_config,
            "importance_score_initialization": self.importance_score_initialization_func,
            "recency_score_initialization": self.recency_score_initialization_func,
            "compound_score_calculation": self.compound_score_calculation_func,
            "decay_function": self.decay_function,
            "importance_score_change_access_counter": self.importance_score_change_access_counter,
            "clean_up_threshold_dict": self.clean_up_threshold_dict,
        }
        with open(os.path.join(save_path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict_to_save, f)
        
        save_universe = {}
        for symbol, data in self.universe.items():
            index_path = os.path.join(save_path, f"{symbol}.index")
            faiss.write_index(data["index"], index_path)
            save_universe[symbol] = {"score_memory": list(data["score_memory"]), "index_save_path": index_path}
        with open(os.path.join(save_path, "universe_index.pkl"), "wb") as f:
            pickle.dump(save_universe, f)

    @classmethod
    def load_checkpoint(cls, path: str, logger: logging.Logger) -> "MemoryDB":
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
        
        state_dict['logger'] = logger
        
        obj = cls(**state_dict)
        
        universe_path = os.path.join(path, "universe_index.pkl")
        if os.path.exists(universe_path):
            with open(universe_path, "rb") as f:
                universe = pickle.load(f)
            for symbol in universe:
                universe[symbol]["index"] = faiss.read_index(universe[symbol]["index_save_path"])
                universe[symbol]["score_memory"] = SortedList(
                    universe[symbol]["score_memory"],
                    key=lambda x: x["important_score_recency_compound_score"]
                )
                del universe[symbol]["index_save_path"]
            obj.universe = universe
        return obj

class BrainDB:
    def __init__(
        self, agent_name: str, emb_config: Dict[str, Any], id_generator: id_generator_func,
        short_term_memory: MemoryDB, mid_term_memory: MemoryDB, long_term_memory: MemoryDB,
        logger: logging.Logger, use_gpu: bool = True,
    ):
        self.agent_name = agent_name
        self.emb_config = emb_config
        self.use_gpu = use_gpu
        self.id_generator = id_generator
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.removed_ids = []
        self.logger = logger

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BrainDB":
        id_generator = id_generator_func()
        agent_name = config["general"]["agent_name"]
        
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            log_path = os.path.join("data", "04_model_output_log", f"{config['general']['trading_symbol']}_run.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            handler = logging.FileHandler(log_path, mode="a")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        emb_config = config["agent"][agent_name]["embedding"]["detail"]
        
        memory_objects = {}
        for layer_key in ["short", "mid", "long"]:
            memory_objects[layer_key] = MemoryDB(
                db_name=f"{agent_name}_{layer_key}",
                id_generator=id_generator, emb_config=emb_config,
                jump_threshold_upper=config[layer_key].get("jump_threshold_upper", 999999) if layer_key != "long" else 999999,
                jump_threshold_lower=config[layer_key].get("jump_threshold_lower", -999999) if layer_key != "short" else -999999,
                importance_score_initialization=get_importance_score_initialization_func(config[layer_key]["importance_score_initialization"], layer_key),
                recency_score_initialization=R_ConstantInitialization(),
                compound_score_calculation=LinearCompoundScore(),
                importance_score_change_access_counter=LinearImportanceScoreChange(),
                decay_function=ExponentialDecay(**config[layer_key]["decay_params"]),
                clean_up_threshold_dict=config[layer_key]["clean_up_threshold_dict"],
                logger=logger,
            )
        
        return cls(
            agent_name=agent_name, id_generator=id_generator, emb_config=emb_config,
            short_term_memory=memory_objects["short"], mid_term_memory=memory_objects["mid"],
            long_term_memory=memory_objects["long"], logger=logger
        )
    
    def add_memory_short(self, symbol: str, date: date, text: str) -> None: self.short_term_memory.add_memory(symbol, date, text)
    def add_memory_mid(self, symbol: str, date: date, text: str) -> None: self.mid_term_memory.add_memory(symbol, date, text)
    def add_memory_long(self, symbol: str, date: date, text: str) -> None: self.long_term_memory.add_memory(symbol, date, text)
    def query_short(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]: return self.short_term_memory.query(query_text, top_k, symbol)
    def query_mid(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]: return self.mid_term_memory.query(query_text, top_k, symbol)
    def query_long(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]: return self.long_term_memory.query(query_text, top_k, symbol)
    
    def update_access_count_with_feed_back(self, symbol: str, ids: Union[List[int], int], feedback: int) -> None:
        if isinstance(ids, int): ids = [ids]
        ids = [i for i in ids if i not in self.removed_ids]
        if not ids: return

        for mem_layer in [self.short_term_memory, self.mid_term_memory, self.long_term_memory]:
            success_ids = mem_layer.update_access_count_with_feed_back(symbol, ids, feedback)
            ids = [i for i in ids if i not in success_ids]
            if not ids: break

    def step(self) -> None:
        for mem in [self.short_term_memory, self.mid_term_memory, self.long_term_memory]:
            self.removed_ids.extend(mem.step())
        
        self.logger.info("Memory jump starts...")
        for _ in range(2):
            jump_up_s, _, del_ids_s = self.short_term_memory.prepare_jump()
            if jump_up_s: self.mid_term_memory.accept_jump((jump_up_s, {}), "up")
            self.removed_ids.extend(del_ids_s)
            
            jump_up_m, jump_down_m, del_ids_m = self.mid_term_memory.prepare_jump()
            if jump_up_m: self.long_term_memory.accept_jump((jump_up_m, {}), "up")
            if jump_down_m: self.short_term_memory.accept_jump(({}, jump_down_m), "down")
            self.removed_ids.extend(del_ids_m)
            
            _, jump_down_l, del_ids_l = self.long_term_memory.prepare_jump()
            if jump_down_l: self.mid_term_memory.accept_jump(({}, jump_down_l), "down")
            self.removed_ids.extend(del_ids_l)
        self.logger.info("Memory jump ends...")

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        if os.path.exists(path) and force: shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump({
                "agent_name": self.agent_name, "emb_config": self.emb_config,
                "removed_ids": self.removed_ids, "id_generator": self.id_generator,
            }, f)
        
        for name, mem in [("short", self.short_term_memory), ("mid", self.mid_term_memory), ("long", self.long_term_memory)]:
            mem.save_checkpoint(name=f"{name}_term_memory", path=path, force=force)

    @classmethod
    def load_checkpoint(cls, path: str):
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
            
        logger = logging.getLogger(__name__)
        
        return cls(
            agent_name=state_dict["agent_name"],
            id_generator=state_dict["id_generator"],
            emb_config=state_dict["emb_config"],
            short_term_memory=MemoryDB.load_checkpoint(os.path.join(path, "short_term_memory"), logger),
            mid_term_memory=MemoryDB.load_checkpoint(os.path.join(path, "mid_term_memory"), logger),
            long_term_memory=MemoryDB.load_checkpoint(os.path.join(path, "long_term_memory"), logger),
            logger=logger,
        )