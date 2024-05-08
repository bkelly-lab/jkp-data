import os
import sys
import urllib.parse
import polars as pl
from wrds import __version__ as wrds_version
from sys import version_info
appname = "{0} python {1}.{2}.{3}/wrds {4}".format(
    sys.platform, version_info[0], version_info[1], version_info[2], wrds_version
)
class Connection(object):
    def __init__(self, autoconnect=True, verbose=False, **kwargs):
        self._verbose = verbose
        self._username = kwargs.get("wrds_username", "")
        self._password = kwargs.get("wrds_password", "")
        self._hostname = "wrds-pgdata.wharton.upenn.edu"
        self._port = 9737
        self._dbname = "wrds"
        self._connect_args = {"sslmode": "require", "application_name": appname}

    def raw_sql_polars_uri(self, sql, 
                           partition_on=None, partition_range=None, 
                           partition_num=None, protocol=None, 
                           engine=None, schema_overrides=None):     
        # Extract the connection details from the class
        username = self._username
        password = urllib.parse.quote_plus(self._password)
        hostname = self._hostname
        port = self._port
        dbname = self._dbname
        connect_args = self._connect_args
        
        # Construct the URI
        uri_args = "&".join([f"{k}={v}" for k, v in connect_args.items()])
        uri = f"postgresql://{username}:{password}@{hostname}:{port}/{dbname}?{uri_args}"
        #Get data using polars.read_database_uri
        df = pl.read_database_uri(
            query=sql,
            uri=uri,
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
            protocol=protocol,
            engine=engine,
            schema_overrides=schema_overrides
        )
        return df
        