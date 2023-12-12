import polars as pl
import wrds_Fernando
wrds_session = wrds_Fernando.Connection(wrds_username= "username", wrds_password="password")
compfunda = wrds_session.raw_sql_polars_uri(f"""
            SELECT *
            FROM comp.funda
            """)
compfunda = compfunda.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
compfunda.write_ipc('compfunda.ft')
del compfunda

compsecd = wrds_session.raw_sql_polars_uri(f"""
            SELECT *
            FROM comp.secd
            """)
compsecd = compsecd.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
compsecd.write_ipc('compsecd.ft')
del compsecd
compgsecd = wrds_session.raw_sql_polars_uri(f"""
            SELECT *
            FROM comp.g_secd
            """)
compgsecd = compgsecd.with_columns([pl.col('gvkey').cast(pl.Int64).alias('gvkey')])
compgsecd.write_ipc('compgsecd.ft')
del compgsecd