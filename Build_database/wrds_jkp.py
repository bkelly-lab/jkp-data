import getpass
import os
import sys
import re
import stat
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import Numeric, Float, DECIMAL, REAL, DOUBLE_PRECISION
import urllib.parse
import polars as pl
from packaging import version
from wrds import __version__ as wrds_version

from sys import version_info

appname = "{0} python {1}.{2}.{3}/wrds {4}".format(
    sys.platform, version_info[0], version_info[1], version_info[2], wrds_version
)

# Sane defaults
WRDS_POSTGRES_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_POSTGRES_PORT = 9737
WRDS_POSTGRES_DB = "wrds"
WRDS_CONNECT_ARGS = {"sslmode": "require", "application_name": appname}

def find_tables(sql):
    pattern = r"(?i)\bFROM\s+([^\s,;]+)|\bJOIN\s+([^\s,;]+)"
    matches = re.findall(pattern, sql)
    table_names = [match[0] or match[1] for match in matches if match[0] or match[1]]
    return table_names

class NotSubscribedError(PermissionError):
    pass


class SchemaNotFoundError(FileNotFoundError):
    pass


class Connection(object):
    def __init__(self, autoconnect=True, verbose=False, **kwargs):
        """
        Set up the connection to the WRDS database.
        By default, also establish the connection to the database.

        Optionally, the user may specify connection parameters:
            *wrds_hostname*: WRDS database hostname
            *wrds_port*: database connection port number
            *wrds_dbname*: WRDS database name
            *wrds_username*: WRDS username
            *wrds_password*: WRDS password
            *autoconnect*: If false will not immediately establish the connection

        The constructor will use the .pgpass file if it exists and may make use of
        PostgreSQL environment variables such as PGHOST, PGUSER, etc., if cooresponding
        parameters are not set.
        If not, it will ask the user for a username and password.
        It will also direct the user to information on setting up .pgpass.

        Additionally, creating the instance will load a list of schemas
        the user has permission to access.

        :return: None

        Usage::
        >>> db = wrds.Connection()
        Loading library list...
        Done
        """
        self._verbose = verbose
        self._username = kwargs.get("wrds_username", "")
        self._password = kwargs.get("wrds_password", "")
        # PGHOST if set will override default for first attempt
        self._hostname = kwargs.get(
            "wrds_hostname", os.environ.get('PGHOST', WRDS_POSTGRES_HOST)
        )
        self._port = kwargs.get("wrds_port", WRDS_POSTGRES_PORT)
        self._dbname = kwargs.get("wrds_dbname", WRDS_POSTGRES_DB)
        self._connect_args = kwargs.get("wrds_connect_args", WRDS_CONNECT_ARGS)

        if autoconnect:
            self.connect()
            self.load_library_list()

    def __make_sa_engine_conn(self, raise_err=False):
        username = self._username
        hostname = self._hostname
        password = urllib.parse.quote_plus(self._password)
        port = self._port
        dbname = self._dbname
        pguri = f"postgresql://{username}:{password}@{hostname}:{port}/{dbname}"
        if self._verbose:
            print(f"postgresql://{username}:@{hostname}:{port}/{dbname}")
        try:
            self.engine = sa.create_engine(
                pguri,
                isolation_level="AUTOCOMMIT",
                connect_args=self._connect_args,
            )
            self.connection = self.engine.connect()
        except Exception as err:
            if self._verbose:
                print(f"{err=}")
            self.engine = None
            if raise_err:
                raise err

    def connect(self):
        """Make a connection to the WRDS database."""
        # first try connection using system defaults and params set in constructor
        self.__make_sa_engine_conn()

        if (self.engine is None and self._hostname != WRDS_POSTGRES_HOST):
            # try explicit w/ default hostname
            print(f"Trying '{WRDS_POSTGRES_HOST}'...")
            self._hostname = WRDS_POSTGRES_HOST
            self.__make_sa_engine_conn()

        if (self.engine is None):
            # Use explicit username and password
            self._username, self._password = self.__get_user_credentials()
            # Last attempt, raise error if Exception encountered
            self.__make_sa_engine_conn(raise_err=True)

            if (self.engine is None):
                print(f"Failed to connect {self._username}@{self._hostname}")
            else:
                # Connection successful. Offer to create a .pgpass for the user.
                print("WRDS recommends setting up a .pgpass file.")
                do_create_pgpass = ""
                while do_create_pgpass != "y" and do_create_pgpass != "n":
                    do_create_pgpass = input("Create .pgpass file now [y/n]?: ")

                if do_create_pgpass == "y":
                    try:
                        self.create_pgpass_file()
                        print("Created .pgpass file successfully.")
                    except Exception:
                        print("Failed to create .pgpass file.")
                print(
                    "You can create this file yourself at any time "
                    "with the create_pgpass_file() function."
                )

    def close(self):
        """
        Close the connection to the database.
        """
        self.connection.close()
        self.engine.dispose()
        self.engine = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def load_library_list(self):
        """Load the list of Postgres schemata (c.f. SAS LIBNAMEs)
        the user has permission to access."""
        self.insp = sa.inspect(self.connection)
        print("Loading library list...")
        query = """
WITH pgobjs AS (
    -- objects we care about - tables, views, foreign tables, partitioned tables
    SELECT oid, relnamespace, relkind
    FROM pg_class
    WHERE relkind = ANY (ARRAY['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"])
),
schemas AS (
    -- schemas we have usage on that represent products
    SELECT nspname AS schemaname,
        pg_namespace.oid,
        array_agg(DISTINCT relkind) AS relkind_a
    FROM pg_namespace
    JOIN pgobjs ON pg_namespace.oid = relnamespace
    WHERE nspname !~ '(^pg_)|(_old$)|(_new$)|(information_schema)'
        AND has_schema_privilege(nspname, 'USAGE') = TRUE
    GROUP BY nspname, pg_namespace.oid
)
SELECT schemaname
FROM schemas
WHERE relkind_a != ARRAY['v'::"char"] -- any schema except only views
UNION
-- schemas w/ views (aka "friendly names") that reference accessable product tables
SELECT nv.schemaname
FROM schemas nv
JOIN pgobjs v ON nv.oid = v.relnamespace AND v.relkind = 'v'::"char"
JOIN pg_depend dv ON v.oid = dv.refobjid AND dv.refclassid = 'pg_class'::regclass::oid
    AND dv.classid = 'pg_rewrite'::regclass::oid AND dv.deptype = 'i'::"char"
JOIN pg_depend dt ON dv.objid = dt.objid AND dv.refobjid <> dt.refobjid
    AND dt.classid = 'pg_rewrite'::regclass::oid
    AND dt.refclassid = 'pg_class'::regclass::oid
JOIN pgobjs t ON dt.refobjid = t.oid
    AND (t.relkind = ANY (ARRAY['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
JOIN schemas nt ON t.relnamespace = nt.oid
GROUP BY nv.schemaname
ORDER BY 1;
        """
        if version.parse(sa.__version__) > version.parse("2"):
            cursor = self.connection.exec_driver_sql(query)
        else:
            cursor = self.connection.execute(query)
        self.schema_perm = [x[0] for x in cursor.fetchall()]
        print("Done")


    def __check_schema_perms(self, schema):
        """
        Check the permissions of the schema.
        Raise permissions error if user does not have access.
        Raise other error if the schema does not exist.

        Else, return True

        :param schema: Postgres schema name.
        :rtype: bool

        """
        if schema in self.schema_perm:
            return True
        else:
            if schema in self.insp.get_schema_names():
                raise NotSubscribedError(
                    "You do not have permission to access "
                    "the {} library".format(schema)
                )
            else:
                raise SchemaNotFoundError("The {} library is not found.".format(schema))

    def list_libraries(self):
        """
        Return all the libraries (schemas) the user can access.

        :rtype: list

        Usage::
        >>> db.list_libraries()
        ['aha', 'audit', 'block', 'boardex', ...]
        """
        return self.schema_perm

    def list_tables(self, library):
        """
        Returns a list of all the views/tables/foreign tables within a schema.

        :param library: Postgres schema name.

        :rtype: list

        Usage::
        >>> db.list_tables('wrdssec')
        ['wciklink_gvkey', 'dforms', 'wciklink_cusip', 'wrds_forms', ...]
        """
        if self.__check_schema_perms(library):
            output = (
                self.insp.get_view_names(schema=library)
                + self.insp.get_table_names(schema=library)
                + self.insp.get_foreign_table_names(schema=library)
            )
            return output

    def __get_schema_for_view(self, schema, table):
        """
        Internal function for getting the schema based on a view
        """
        sql_code = """SELECT distinct(source_ns.nspname) AS source_schema
                      FROM pg_depend
                      JOIN pg_rewrite
                        ON pg_depend.objid = pg_rewrite.oid
                      JOIN pg_class as dependent_view
                        ON pg_rewrite.ev_class = dependent_view.oid
                      JOIN pg_class as source_table
                        ON pg_depend.refobjid = source_table.oid
                      JOIN pg_attribute
                        ON pg_depend.refobjid = pg_attribute.attrelid
                          AND pg_depend.refobjsubid = pg_attribute.attnum
                      JOIN pg_namespace dependent_ns
                        ON dependent_ns.oid = dependent_view.relnamespace
                      JOIN pg_namespace source_ns
                        ON source_ns.oid = source_table.relnamespace
                      WHERE dependent_ns.nspname = '{schema}'
                        AND dependent_view.relname = '{view}';
                    """.format(
            schema=schema, view=table
        )
        if self.__check_schema_perms(schema):
            if version.parse(sa.__version__) > version.parse("2"):
                result = self.connection.exec_driver_sql(sql_code)
            else:
                result = self.connection.execute(sql_code)
            return result.fetchone()[0]

    def describe_table(self, library, table):
        """
        Takes the library and the table and describes all the columns
        in that table.
        Includes Column Name, Column Type, Nullable?, Comment

        :param library: Postgres schema name.
        :param table: Postgres table name.

        :rtype: polars.DataFrame

        Usage::
        >>> db.describe_table('wrdssec_all', 'dforms')
                    name nullable     type comment
              0      cik     true  VARCHAR
              1    fdate     true     DATE
              2  secdate     true     DATE
              3     form     true  VARCHAR
              4   coname     true  VARCHAR
              5    fname     true  VARCHAR
        """
        rows = self.get_row_count(library, table)
        # print("Approximately {} rows in {}.{}.".format(rows, library, table))
        table_info_dict = self.insp.get_columns(table, schema=library)
        table_info = pl.DataFrame(table_info_dict)
        return table_info.select(['name', 'nullable', 'type', 'comment'])

    def get_row_count(self, library, table):
        """
        Uses the library and table to get the approximate row count for the table.

        :param library: Postgres schema name.
        :param table: Postgres table name.

        :rtype: int

        Usage::
        >>> db.get_row_count('wrdssec', 'dforms')
        16378400
        """

        sqlstmt = """
            EXPLAIN (FORMAT 'json')  SELECT 1 FROM {}.{} ;
        """.format(
            sa.sql.quoted_name(library, True), sa.sql.quoted_name(table, True)
        )

        try:
            if version.parse(sa.__version__) > version.parse("2"):
                result = self.connection.exec_driver_sql(sqlstmt)
            else:
                result = self.connection.execute(sqlstmt)
            return int(result.fetchone()[0][0]["Plan"]["Plan Rows"])
        except Exception as e:
            print("There was a problem with retrieving the row count: {}".format(e))
            return 0

    def raw_sql(
        self,
        sql,
        iter_batches=False,
        batch_size=None,
        schema_overrides={},
        infer_schema_length=None,
    ):
        """
        Queries the database using a raw SQL string.

        :param sql: SQL code in string object.
        :param coerce_float: (optional) boolean, default: True
            Attempt to convert values to non-string, non-numeric objects
            to floating point. Can result in loss of precision.
        :param date_cols: (optional) list or dict, default: None
            - List of column names to parse as date
            - Dict of ``{column_name: format string}`` where
                format string is:
                  strftime compatible in case of parsing string times or
                  is one of (D, s, ns, ms, us) in case of parsing
                    integer timestamps
            - Dict of ``{column_name: arg dict}``,
                where the arg dict corresponds to the keyword arguments of
                  :func:`pandas.to_datetime`
        :param index_col: (optional) string or list of strings,
          default: None
            Column(s) to set as index(MultiIndex)
        :param params: parameters to SQL query, if parameterized.
        :param chunksize: (optional) integer or None default: 500000
            Process query in chunks of this size. Smaller chunksizes can save
            a considerable amount of memory while query is being processed.
            Set to None run query w/o chunking.
        :param return_iter: (optional) boolean, default:False
            When chunksize is not None, return an iterator where chunksize
            number of rows is included in each chunk.
        :param dtype_backend: (optional) string
          default: "numpy_nullable"
            Allow backend storage type to be changed. e.g. "pyarrow"

        :rtype: pandas.DataFrame or or Iterator[pandas.DataFrame]


        Usage ::
        # Basic Usage
        >>> data = db.raw_sql('select cik, fdate, coname from wrdssec_all.dforms;', date_cols=['fdate'], index_col='cik')
        >>> data.head()
            cik        fdate       coname
            0000000003 1995-02-15  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1996-02-14  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1997-02-19  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1998-03-02  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1998-03-10  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y..
            ...

        # Parameterized SQL query
        >>> parm = {'syms': ('A', 'AA', 'AAPL'), 'num_shares': 50000}
        >>> data = db.raw_sql('select * from taqmsec.ctm_20030910 where sym_root in %(syms)s and size > %(num_shares)s', params=parm)
        >>> data.head()
                  date           time_m ex sym_root sym_suffix tr_scond      size   price tr_stopind tr_corr     tr_seqnum tr_source tr_rf
            2003-09-10  11:02:09.485000  T        A       None     None  211400.0  25.350          N      00  1.929952e+15         C  None
            2003-09-10  11:04:29.508000  N        A       None     None   55500.0  25.180          N      00  1.929952e+15         C  None
            2003-09-10  15:08:21.155000  N        A       None     None   50500.0  24.470          N      00  1.929967e+15         C  None
            2003-09-10  16:10:35.522000  T        A       None        B   71900.0  24.918          N      00  1.929970e+15         C  None
            2003-09-10  09:35:20.709000  N       AA       None     None  108100.0  28.200          N      00  1.929947e+15         C  None
        """  # noqa

        try:
            
            for name in find_tables(sql):
                lib, table = name.split('.')
                dtypes = self.describe_table(lib, table).select(['name', 'type']).to_numpy()
                for i in dtypes:
                    if isinstance(i[1], (Numeric, Float, DECIMAL, REAL, DOUBLE_PRECISION)): schema_overrides[i[0]] = pl.Float64
            if not schema_overrides: schema_overrides = None
                
            df = pl.read_database(
                sql,
                self.connection,
                iter_batches=iter_batches,
                batch_size=batch_size,
                schema_overrides=schema_overrides,
                infer_schema_length=infer_schema_length
            )
            if iter_batches or batch_size is None: return df
            else: return pl.concat(list(df))
        except sa.exc.ProgrammingError as e:
            raise e
