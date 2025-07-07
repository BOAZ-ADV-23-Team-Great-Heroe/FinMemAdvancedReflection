import os
import numpy as np
from typing import List, Union
from langchain_community.embeddings import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

class OpenAILongerThanContextEmb:
    """
    Embedding function with openai as embedding backend.
    If the input is larger than the context size, the input is split into chunks of size `chunk_size` and embedded separately.
    The final embedding is the average of the embeddings of the chunks.
    Details see: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    """

    def __init__(
        self,
        openai_api_key: Union[str, None] = None,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 5000,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the Embedding object.

        Args:
            openai_api_key (str): The API key for OpenAI.
            embedding_model (str, optional): The model to use for embedding. Defaults to "text-embedding-ada-002".
            chunk_size (int, optional): The maximum number of token to send to openai embedding model at one time. Defaults to 5000.
            verbose (bool, optional): Whether to show progress bar during embedding. Defaults to False.

        Returns:
            None
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        logger.info(f"Using embedding model: {embedding_model}")
        logger.info(f"Chunk size: {chunk_size}")
        logger.info(f"API Key exists: {bool(self.openai_api_key)}")
        
        self.emb_model = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            chunk_size=chunk_size,
            show_progress_bar=verbose,
        )

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        Asynchronously performs embedding on a list of text.

        This method calls the `aembed_documents` method of the `emb_model` object to embed the input text.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            List[List[float]]: The embeddings of the input text as a list of lists of floats.

        """
        if isinstance(text, str):
            text = [text]
        
        logger.info(f"Input text type: {type(text)}")
        logger.info(f"Input text length: {len(text)}")
        logger.info(f"First text sample: {text[0][:100] if text else 'Empty'}")
        
        try:
            logger.info("Starting embedding process...")
            logger.info(f"Using model: {self.emb_model.model}")
            logger.info(f"API Key exists: {bool(self.openai_api_key)}")
            
            result = self.emb_model.embed_documents(texts=text, chunk_size=None)
            logger.info(f"Embedding result shape: {np.array(result).shape}")
            logger.info(f"First embedding sample: {result[0][:5] if result else 'Empty'}")
            return result
        except Exception as e:
            logger.error(f"Error during embedding: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            raise

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        Performs embedding on a list of text.

        This method calls the `_emb` method to asynchronously embed the input text using the `emb_model` object.

        Args:
            self: The instance of the class.
            text (List[str]): A list of text to be embedded.

        Returns:
            np.array: The embedding of the input text as a NumPy array.

        """
        try:
            logger.info("Starting __call__ method...")
            result = np.array(self._emb(text)).astype("float32")
            logger.info(f"Final embedding shape: {result.shape}")
            logger.info(f"Final embedding type: {result.dtype}")
            logger.info(f"First embedding sample: {result[0][:5] if result.size > 0 else 'Empty'}")
            return result
        except Exception as e:
            logger.error(f"Error in __call__: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            raise

    def get_embedding_dimension(self):
        """
        Returns the dimension of the embedding.

        This method checks the value of `self.emb_model.model` and returns the corresponding embedding dimension. If the model is not implemented, a `NotImplementedError` is raised.

        Args:
            self: The instance of the class.

        Returns:
            int: The dimension of the embedding.

        Raises:
            NotImplementedError: Raised when the embedding dimension for the specified model is not implemented.

        """
        if self.emb_model.model == "text-embedding-ada-002":
            return 1536
        else:
            raise NotImplementedError(
                f"Embedding dimension for model {self.emb_model.model} not implemented"
            )
