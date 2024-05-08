Hello user!
To run the code, make sure that the following 2 files can be executed: 'call_roll_apply_daily.sh' and 'run_code.sh'
You can do this by running the following commands:
'chmod +x call_roll_apply_daily.sh'
'chmod +X run_code.sh'

Fill out the line:
" wrds_session = wrds_Fernando.Connection(wrds_username= "your_username_here", wrds_password="your_password_here")"
in the trial_full_run.py and WRDS_SQL_queries_Fernando.py files with your WRDS user data. 

At the beginning of the execution of the code, DUO may send an alert to your phone to grant access to download the data.
If the login attemps time out because you didn't grant access with DUO, stop the code and start again.

That's it, we'll take it from there.
I'm running the code on 512 gb of RAM and 128 CPU cores.
