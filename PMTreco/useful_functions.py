import midas.file_reader

from cygno import s3, cmd



def getruns(runs, path = './tmp/', tag = 'LNGS', cloud = True, verbose = False):
    for r in runs:
        
        fname = s3.mid_file(r, tag=tag, cloud=cloud, verbose=verbose)
        
        filetmp = cmd.cache_file(fname, cachedir=path, verbose=verbose)

