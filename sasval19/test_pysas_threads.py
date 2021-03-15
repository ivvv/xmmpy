#
# testing pysas command wrappers
#%%
import os
import shutil
import glob
import tempfile
import concurrent.futures

from pysas.sasver import sasver
from pysas.sastask import MyTask as task

#%%

def prepare_sas_task(command,params=[]):
    #
    #
    try:
        t = task(command,params)
        t.readparfile()
        t.processargs()
    except:
        print (f'Preparing task {command} with parameters {params} raised an exception.')
        return None
    return t
#%%
#
#
def run_sas_task(xtask):
    #
    # each task will be run in temporary folder
    # upon successful completion, all files will be copied to the upper folder, before the temporary folder and all of its content is destroyed
    #
    cwd = os.getcwd()
    if (isinstance(xtask,task)):
        #
        #
        # process in a temporary subfolder
        with tempfile.TemporaryDirectory() as tmpdir:
            print('created temporary directory', tmpdir)
            os.chdir(tempdir)
            # have to use try: except: because the current pysas implementation does not provide execution status
            try:
                xtask.runtask()
            except:
                print ('Task execution failed')
                os.chdir(cwd)
                return None            
            #
            # hopefully the processing is done, now we have to copy back the files
            #
            for file in os.listdir():
                if (os.path.isfile(file)):
                    shutil.copy(file, cwd)
                else:
                    print (f'{file} is not a file, cannot copy')
            pass
        # the temdir will be closed and all content removed after this point
        pass
    else:
        return None
    return 0
#%%
sx = prepare_sas_task('startsas',params=['odfid=0727780501'])
status = run_sas_task(sx)

# %%
#
# prepare all the tasks to run in multi-threading
#
p = prepare_sas_task('epproc')
#
m = prepare_sas_task('emproc')
#
r = prepare_sas_task('rgsproc')
#
o = prepare_sas_task('omichain')
#
#%%
#
# now the multi-threading part
#
out_dir = os.path.join(os.getcwd(),'output')
if (not os.path.isdir(out_dir)):
    os.mkdir(out_dir)
#
# do all processing in a dedicated folder
#
os.chdir(out_dir)
#
# will hold the status and thread output of each command 
futures_list = []
#
tasks_list = (p,m,r,o)
#
#%%
#
# use one possibility to do multi-threading, there are alternatives
#
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for x in tasks_list:
        futures = executor.submit(run_sas_task,x)
        futures_list.append(futures)
#
print ("*** All done ** ")
for ff in futures_list:
    #results = ff
    print (ff,'Done: ',ff.done(),'; Status: ',ff.result())
