import os
import io
import json
import hashlib
import tempfile
from typing import Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import redis
from zoneinfo import ZoneInfo

class MemoryManager():
    def __init__(self, redis_url: str = "redis://localhost:6379/0", storage_dir: str = "./df_states",timezone: str = "Europe/Moscow"):
        """The constructor of the class in which the connection to the redis database is set, by default it is localhost
        As well as the folder where df will be stored locally on the system.

        Args:
            redis_url (_type_, optional): _description_. Defaults to "redis://localhost:6379/0".
            storage_dir (str, optional): _description_. Defaults to "./df_states".
            tz (str, optional): Variable for choosing timezone for metadata. Defaults to Europe/Moscow.
        """
        self.r = redis.from_url(redis_url, decode_responses=True)
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.tz = ZoneInfo(timezone)
    def _state_filename(self,session_id: str, step: int) -> str:
        """Generating a file name to save its state

        Args:
            session_id (str): id сессии
            step (int): шаг в системе

        Returns:
            str: a string with the session name, as well as a step in the .parquet format
        """
        return os.path.join(self.storage_dir,f"{session_id}_state_{step}.parquet")
    @staticmethod
    def df_describtion(df:pd.DataFrame, sample_n: int = 5) -> str:
        """Function for creating hash of dataframe to save current df hash.

        Args:
            df (pd.DataFrame): takes df as pd.dataframe
            sample_n (int, optional): amount of rows including to sample function. Defaults to 5.

        Returns:
            str: hash of dataframe
        """
        cols = ",".join(map(str, df.columns.tolist()))
        dtypes = ",".join(str(x) for x in list(df.dtypes))
        shape = f"{df.shape[0]}x{df.shape[1]}"
        sample = df.head(sample_n).to_csv()
        raw = "||".join([cols,dtypes,shape,sample])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    def init_session_from_csv(self,session_id: str, csv_path: str) -> dict:
        """Launch init session from csv file

        Args:
            session_id (str): id of session
            csv_path (str): path to current csv

        Returns:
            dict: dictionary with session information and note about init
        """
        df = pd.read_csv(csv_path)
        return self._push_new_state(session_id, df,note = f"init_from:{os.path.basename(csv_path)}")
    def _push_new_state(self,session_id:str,df:pd.DataFrame,note: Optional[str] = None) -> dict:
        first_key = f"session:{session_id}:states"
        idx_key = f"session:{session_id}:idx"
        meta_key = f"session:{session_id}:meta"

        #Pipilene for avoiding race between clients
        pipe = self.r.pipeline()
        pipe.llen(first_key)
        length = pipe.execute()[0]
        new_index = length
        
        #Saving df

        filename = self._state_filename(session_id,new_index)
        df.to_parquet(filename,index=False)

        #Compute df description

        descript = self.df_describtion(df)

        pipe = self.r.pipeline()
        pipe.rpush(first_key,filename)
        pipe.set(idx_key,new_index)
        #storing metadata
        current_tz = datetime.now(self.tz)
        meta = {
            "created_at": current_tz.isoformat(),
            "index": new_index,
            "signature/df_description": descript,
            "note": note or "",
            "nrows": df.shape[0],
            "ncols": df.shape[1]
        }
        pipe.hset(meta_key,mapping=meta)
        pipe.execute()
        return meta
    def get_current_state_info(self, session_id: str) -> Optional[dict]:
        """Function for getting current info about state in redis db

        Args:
            session_id (str): id of needed session

        Returns:
            Optional[dict]: meta dict or empty dict
        """
        first_key = f"session:{session_id}:states"
        idx_key = f"session:{session_id}:idx"
        length = self.r.llen(first_key)
        if length == 0:
            return None
        idx = int(self.r.get(idx_key) or 0)
        idx = max(0, min(idx,length - 1))
        filename = self.r.lindex(first_key, idx)
        meta_key = f"session:{session_id}:meta"
        meta = self.r.hgetall(meta_key) or {}
        meta.update({"index":idx,
                     "filename":filename})
        return meta
    def load_current_df(self,session_id: str) -> Optional[pd.DataFrame]:
        """Loading current dataframe by session_id

        Args:
            session_id (str): session id

        Raises:
            FileNotFoundError: if no file founded on system

        Returns:
            Optional[pd.DataFrame]: returns dataframe from .parquet
        """
        info = self.get_current_state_info(session_id)
        if not info:
            return None
        filename = info.get("filename")
        if not filename or not os.path.exists(filename):
            raise FileNotFoundError(f"State file missing: {filename}")
        return pd.read_parquet(filename)
    def push_result(self,session_id: str, df_new: pd.DataFrame, code: str = '', note: Optional[str] = None)-> dict:
        """Calling function after succesfull llm code launch and checks for changes in df

        Args:
            session_id (str): id of current session
            df_new (pd.DataFrame): dataframe after executed llm code
            code (str, optional): executed code Defaults to ''.
            note (_type_, optional): optional note for llm code. Defaults to Optional[str]=None.

        Returns:
            dict: updated meta data with structure changed bool and llm_code str
        """

        df_before = self.load_current_df(session_id)
        cols_before = set(df_before.columns) if df_before is not None else set()
        cols_after = set(df_new.columns)

        meta = self._push_new_state(session_id, df_new, note=note)

        structure_changed = cols_before != cols_after

        meta["structure_changed"] = structure_changed
        meta["llm_code"] = code
        return meta
    def undo(self, session_id: str) -> Optional[dict]:
        """unfo function for user to undo what he did, changing index to select previous meta data

        Args:
            session_id (str): session id

        Returns:
            Optional[dict]: current state by idx info
        """
        first_key = f"session:{session_id}:states"
        idx_key = f"session:{session_id}:idx"
        length = self.r.llen(first_key)
        if length == 0:
            return None
        idx = int(self.r.get(idx_key) or 0)
        if idx <= 0:
            return self.get_current_state_info(session_id)
        new_idx = idx - 1
        self.r.set(idx_key,new_idx)
        return self.get_current_state_info(session_id)
    def redo(self,session_id: str) -> Optional[dict]:
        first_key = f"session:{session_id}:states"
        idx_key = f"session:{session_id}:idx"
        length = self.r.llen(first_key)
        if length == 0:
            return None
        idx = int(self.r.get(idx_key) or 0)
        if idx >= length - 1:
            return self.get_current_state_info(session_id)
        new_idx = idx + 1
        self.r.set(idx_key, new_idx)
        return self.get_current_state_info(session_id)
    @staticmethod
    def _cache_key(prompt_template: str, df_description: str, user_query: str) -> str:
        """static function for caching info for redis. Prompt, info about df and users query

        Args:
            prompt_template (str) current prompt
            df_description (str) info about dataframe from redis
            user_query (str) users query

        Returns:
            str: hash of all that combain info
        """
        s = "||".join([prompt_template,df_description,user_query])
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    def get_cache(self, prompt_template: str, df_description: str, user_query: str,ttl_seconds: int = 60 * 60 * 24) -> Optional[dict]:
        key = self._cache_key(prompt_template,df_description)
        redis_key = f"llmcache:{key}"
        payload = dict(payload)
        current_tz = datetime.now(self.tz)
        payload.setdefault("created_at",current_tz.isoformat())
        self.r.set(redis_key, json.dumps(payload),ex=ttl_seconds)
    