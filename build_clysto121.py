"""Build a clearly named Clysto-evaluation derivative, NOT original Clysto.
Strict 121 geometry/history/PVS are reused from our independently tested kernel.
The upstream hand-written positional table is mapped to the central rhombus;
side-wing coordinates clamp to the nearest table edge. No 81 games/models/book.
"""
import hashlib,json,os,re,shlex,subprocess,sys,sysconfig
from pathlib import Path
import rules121 as r
ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('CLYSTO_DIR','/Users/liuzirui/Desktop/py/clysto-cc'))/'src/game.hpp'

def build():
    text=SOURCE.read_text()
    raw=re.search(r'PIECE_SCORE_TABLE\[81\] = \{(.*?)\};',text,re.S).group(1)
    raw=re.sub(r'//[^\n]*','',raw);scores=list(map(int,re.findall(r'\d+',raw)))
    if len(scores)!=81:raise ValueError('upstream score table changed')
    table=[]
    for q,y in r.COORDS:
        x=4-q;z=y+4+q
        table.append(scores[min(8,max(0,z))*9+min(8,max(0,x))])
    generated=ROOT/'clysto121_generated';generated.mkdir(exist_ok=True)
    (generated/'upstream_game.hpp').write_text(text)
    manifest=dict(upstream=str(SOURCE),upstream_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  kernel_sha256=hashlib.sha256((ROOT/'ccore121.c').read_bytes()).hexdigest(),
                  table81=scores,table121=table,distance121=[y+8 for q,y in r.COORDS],
                  scope='Clysto positional evaluator derivative + our strict121 PVS. Not original Clysto; no opening book/TT/null pruning.',
                  wing_rule='x=4-q,y=r+4+q; clamp each table coordinate to [0,8]')
    (generated/'manifest.json').write_text(json.dumps(manifest,indent=2))
    evaluator='''static const int CLY_SCORE[121]={SCORES};
static const int CLY_PROGRESS[121]={DISTANCES};
static int evaluate(Board b,int p) {
    int score[3]={0},last[3]={1000000,1000000,1000000};
    for(int i=0;i<121;i++) for(int side=1;side<=2;side++) if(has(b.p[side],i)) {
        int k=side==1?i:120-i;
        score[side]+=CLY_SCORE[k];
        if(CLY_PROGRESS[k]<last[side])last[side]=CLY_PROGRESS[k];
    }
    for(int side=1;side<=2;side++)score[side]-=1 << (last[side]<4?4-last[side]:0);
    /* Match upstream static evaluator; strict terminal is checked separately. */
    if(last[2]==13){score[2]=10000;score[1]=0;}
    if(last[1]==13){score[1]=10000;score[2]=0;}
    return score[p]-score[3-p];
}
'''.replace('SCORES',','.join(map(str,table))).replace('DISTANCES',','.join(str(y+8) for q,y in r.COORDS))
    kernel=(ROOT/'ccore121.c').read_text()
    start=kernel.index('static int evaluate(');end=kernel.index('/* -1 ongoing;',start)
    kernel=kernel[:start]+evaluator+kernel[end:]
    kernel=kernel.replace('"ccore121"','"clysto121_native"').replace('PyInit_ccore121','PyInit_clysto121_native')
    kernel='/* Generated Clysto-evaluator derivative; see manifest.json. */\n'+kernel
    path=generated/'core.c';path.write_text(kernel)
    output=ROOT/('clysto121_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    cmd=shlex.split(os.environ.get('CC','clang'))+['-O3','-Wall','-Wextra','-Wno-unused-parameter','-fPIC','-shared','-I'+sysconfig.get_path('include'),'-I'+str(ROOT)]
    if sys.platform=='darwin':cmd+=['-undefined','dynamic_lookup']
    subprocess.run(cmd+[str(path),'-o',str(output)],check=True)
    print(output)
if __name__=='__main__':build()
