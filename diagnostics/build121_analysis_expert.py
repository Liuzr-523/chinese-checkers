"""Build the frozen diagnostic expert using the current Python's C headers."""
import argparse,os,shlex,subprocess,sys,sysconfig
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out-dir',type=Path,default=ROOT);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);output=a.out_dir/('clysto121_native'+sysconfig.get_config_var('EXT_SUFFIX'))
 cmd=shlex.split(os.environ.get('CC','clang'))+['-O3','-fPIC','-shared','-I'+sysconfig.get_path('include'),'-I'+str(ROOT)]
 if sys.platform=='darwin':cmd+=['-undefined','dynamic_lookup']
 subprocess.run(cmd+[str(ROOT/'clysto121_generated/core.c'),'-o',str(output)],check=True);print(output)
if __name__=='__main__':main()
