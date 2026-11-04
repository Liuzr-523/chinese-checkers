/* Independent strict 121-cell PVS. No 81-cell evaluation/data is reused.
 * 128-bit occupancy, goal-lock reachability, exact repetition history,
 * stable evaluation ordering, iterative deepening, monotonic deadline.
 * TT cutoffs/null moves are deliberately disabled: history and equal-turn
 * semantics must not be approximated by the old 81-cell search shortcuts.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ccore121_tables.h"
typedef unsigned __int128 Bits;
typedef struct { Bits p[3]; } Board;
typedef struct { int s,e,score; } Move;
typedef struct { Board b; int player,count; } Record;
#define CAP 1210
#define INF 1000000
#define WIN 100000
static Bits bit(int i) { return (Bits)1 << i; }
static int has(Bits b,int i) { return (b & bit(i)) != 0; }
static int goal(int p,int i) { return p==1 ? i>=111 : i<10; }
static Board moved(Board b, Move m,int p) { b.p[p] ^= bit(m.s)|bit(m.e); return b; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int same(Board a,Board b) { return a.p[1]==b.p[1] && a.p[2]==b.p[2]; }
static int generate(Board b,int p,Move *out) {
    int n=0;
    for(int s=0;s<121;s++) if(has(b.p[p],s)) {
        Bits occupied=(b.p[1]|b.p[2]) & ~bit(s), seen=bit(s);
        int stack[121],size=1; stack[0]=s;
        while(size) {
            int here=stack[--size];
            for(int k=0;k<6;k++) {
                int mid=NEIGHBOR[here][k], dst=HOP[here][k];
                if(mid<0 || dst<0 || !has(occupied,mid) || has(occupied|seen,dst)) continue;
                if(goal(p,here) && !goal(p,dst)) continue;
                seen |= bit(dst); stack[size++]=dst;
                out[n++]=(Move){s,dst,0};
            }
        }
        for(int k=0;k<6;k++) {
            int dst=NEIGHBOR[s][k];
            if(dst>=0 && !has(occupied|seen,dst) && (!goal(p,s)||goal(p,dst))) {
                seen |= bit(dst); out[n++]=(Move){s,dst,0};
            }
        }
    }
    return n;
}
static int evaluate(Board b,int p) {
    int cost[3]={0}, far[3]={0};
    for(int i=0;i<121;i++) for(int side=1;side<=2;side++) if(has(b.p[side],i)) {
        int d=DIST[i][side-1]; cost[side]+=d; if(d>far[side]) far[side]=d;
    }
    return cost[3-p]+2*far[3-p]-cost[p]-2*far[p];
}
/* -1 ongoing; 0 draw; 1/2 winner. Repetition and ply checked in context. */
static int terminal(Board b,int p) {
    int own[3]={0}, full[3]={0};
    for(int side=1;side<=2;side++) for(int i=0;i<121;i++) if(goal(side,i)) {
        own[side]+=has(b.p[side],i); full[side]+=has(b.p[1]|b.p[2],i);
    }
    if(own[1]==10) {
        if(own[2]==10) return 0;
        if(p==2 && own[2]==9) {
            Move moves[CAP]; int n=generate(b,2,moves);
            for(int i=0;i<n;i++) if(!goal(2,moves[i].s)&&goal(2,moves[i].e)) return 0;
        }
        return 1;
    }
    if(own[2]==10) return 2;
    for(int side=1;side<=2;side++) if(own[side]>0 && full[side]==10) return side;
    return -1;
}
typedef struct {
    double deadline; int aborted,max_ply,base_ply;
    long long nodes,prunes;
    Record *history; Py_ssize_t nhistory;
    Board branch[64]; int turns[64];
    Move workspace[64][CAP]; /* Heap-owned: safe on small GUI worker stacks. */
} Context;
static int outcome(Board b,int p,int level,Context *c) {
    int win=terminal(b,p); if(win>=0) return win;
    int repeats=0;
    for(Py_ssize_t i=0;i<c->nhistory;i++) if(c->history[i].player==p && same(c->history[i].b,b)) repeats+=c->history[i].count;
    /* Root already occurs in the supplied history. Only count child arrivals. */
    for(int i=1;i<=level;i++) if(c->turns[i]==p && same(c->branch[i],b)) repeats++;
    if(repeats>=3 || (c->max_ply>0 && c->base_ply+level>=c->max_ply)) return 0;
    return -1;
}
static void order(Board b,int p,Move *moves,int n) {
    for(int i=0;i<n;i++) moves[i].score=evaluate(moved(b,moves[i],p),p);
    for(int i=1;i<n;i++) {
        Move m=moves[i]; int j=i;
        while(j>0 && m.score>moves[j-1].score) { moves[j]=moves[j-1]; j--; }
        moves[j]=m;
    }
}
static int pvs(Board b,int p,int depth,int alpha,int beta,int level,Context *c) {
    c->nodes++;
    if(now()>=c->deadline) { c->aborted=1; return 0; }
    c->branch[level]=b; c->turns[level]=p;
    int win=outcome(b,p,level,c);
    if(win>=0) return win==0?0:(win==p?WIN:-WIN);
    Move *moves=c->workspace[level]; int n=generate(b,p,moves);
    if(!n) return 0;
    if(depth==0) return evaluate(b,p);
    order(b,p,moves,n);
    int best=-INF;
    for(int i=0;i<n;i++) {
        Board child=moved(b,moves[i],p);
        int value;
        if(i==0) value=-pvs(child,3-p,depth-1,-beta,-alpha,level+1,c);
        else {
            value=-pvs(child,3-p,depth-1,-alpha-1,-alpha,level+1,c);
            if(!c->aborted && value>alpha && value<beta)
                value=-pvs(child,3-p,depth-1,-beta,-alpha,level+1,c);
        }
        if(c->aborted) return 0;
        if(value>best) best=value;
        if(value>alpha) alpha=value;
        if(alpha>=beta) { c->prunes++; break; }
    }
    return best;
}
static int parse_board(PyObject *obj,Board *b) {
    PyObject *seq=PySequence_Fast(obj,"board must be a sequence"); if(!seq) return 0;
    *b=(Board){{0}}; int count[3]={0};
    if(PySequence_Fast_GET_SIZE(seq)!=121) { PyErr_SetString(PyExc_ValueError,"board must have 121 cells"); Py_DECREF(seq); return 0; }
    for(int i=0;i<121;i++) {
        long v=PyLong_AsLong(PySequence_Fast_GET_ITEM(seq,i));
        if(PyErr_Occurred()) { Py_DECREF(seq); return 0; }
        if(v<0||v>2) { PyErr_SetString(PyExc_ValueError,"invalid piece"); Py_DECREF(seq); return 0; }
        if(v) { b->p[v]|=bit(i); count[v]++; }
    }
    Py_DECREF(seq);
    if(count[1]!=10||count[2]!=10) { PyErr_SetString(PyExc_ValueError,"need ten pieces per side"); return 0; }
    return 1;
}
static int valid_player(int p) { if(p==1||p==2) return 1; PyErr_SetString(PyExc_ValueError,"player must be 1 or 2"); return 0; }
static PyObject *py_moves(PyObject *self,PyObject *args) {
    PyObject *obj; int p; Board b;
    if(!PyArg_ParseTuple(args,"Oi",&obj,&p)||!valid_player(p)||!parse_board(obj,&b)) return NULL;
    Move moves[CAP]; int n=generate(b,p,moves); PyObject *result=PyList_New(n); if(!result) return NULL;
    for(int i=0;i<n;i++) { PyObject *m=Py_BuildValue("(ii)",moves[i].s,moves[i].e); if(!m) { Py_DECREF(result); return NULL; } PyList_SET_ITEM(result,i,m); }
    return result;
}
static PyObject *py_eval(PyObject *self,PyObject *args) {
    PyObject *obj; int p; Board b;
    if(!PyArg_ParseTuple(args,"Oi",&obj,&p)||!valid_player(p)||!parse_board(obj,&b)) return NULL;
    return PyLong_FromLong(evaluate(b,p));
}
static PyObject *py_terminal(PyObject *self,PyObject *args) {
    PyObject *obj; int p; Board b;
    if(!PyArg_ParseTuple(args,"Oi",&obj,&p)||!valid_player(p)||!parse_board(obj,&b)) return NULL;
    return PyLong_FromLong(terminal(b,p));
}
static PyObject *py_search(PyObject *self,PyObject *args) {
    PyObject *obj,*hist; int p,depth,ply,max_ply=240; double seconds; Board b;
    if(!PyArg_ParseTuple(args,"OidiiO|i",&obj,&p,&seconds,&depth,&ply,&hist,&max_ply)||!valid_player(p)||!parse_board(obj,&b)) return NULL;
    if(depth<1||depth>62||!isfinite(seconds)||seconds<=0||ply<0||max_ply<0) { PyErr_SetString(PyExc_ValueError,"invalid search limits"); return NULL; }
    PyObject *seq=PySequence_Fast(hist,"history must be (board,player,count) records"); if(!seq) return NULL;
    Context *c=calloc(1,sizeof(Context)); if(!c) { Py_DECREF(seq); return PyErr_NoMemory(); }
    c->nhistory=PySequence_Fast_GET_SIZE(seq);
    c->history=calloc((size_t)c->nhistory+1,sizeof(Record));
    if(!c->history) { free(c); Py_DECREF(seq); return PyErr_NoMemory(); }
    int root_count=0;
    for(Py_ssize_t i=0;i<c->nhistory;i++) {
        PyObject *row=PySequence_Fast_GET_ITEM(seq,i),*hb; int hp,hc;
        if(!PyArg_ParseTuple(row,"Oii",&hb,&hp,&hc)||!valid_player(hp)||!parse_board(hb,&c->history[i].b)) goto fail;
        if(hc<1||hc>3) { PyErr_SetString(PyExc_ValueError,"history count must be 1..3"); goto fail; }
        c->history[i].player=hp; c->history[i].count=hc;
        if(hp==p && same(b,c->history[i].b)) root_count+=hc;
    }
    if(!root_count) c->history[c->nhistory++]=(Record){b,p,1};
    Py_DECREF(seq); seq=NULL;
    c->deadline=now()+seconds; c->base_ply=ply; c->max_ply=max_ply;
    Move moves[CAP],best={-1,-1,0}; int reached=0,score=evaluate(b,p);
    Py_BEGIN_ALLOW_THREADS
    int win=outcome(b,p,0,c);
    if(win>=0) score=win==0?0:(win==p?WIN:-WIN);
    else {
        int n=generate(b,p,moves); order(b,p,moves,n);
        if(!n) score=0;
        else {
            best=moves[0];
            for(int d=1;d<=depth;d++) {
                int value=-INF; Move candidate=best;
                for(int i=0;i<n;i++) {
                    int v=-pvs(moved(b,moves[i],p),3-p,d-1,-INF,-value,1,c);
                    if(c->aborted) break;
                    if(v>value) { value=v; candidate=moves[i]; }
                }
                if(c->aborted) break;
                best=candidate; score=value; reached=d;
                /* Stable promotion of completed principal variation. */
                for(int i=0;i<n;i++) if(moves[i].s==best.s && moves[i].e==best.e) {
                    for(int j=i;j>0;j--) moves[j]=moves[j-1]; moves[0]=best; break;
                }
            }
        }
    }
    Py_END_ALLOW_THREADS
    PyObject *result=Py_BuildValue("(iiiiLLi)",best.s,best.e,score,reached,c->nodes,c->prunes,c->aborted);
    free(c->history); free(c); return result;
fail:
    Py_XDECREF(seq); free(c->history); free(c); return NULL;
}
static PyMethodDef methods[]={
 {"moves",py_moves,METH_VARARGS,"Legal endpoints"},{"evaluate",py_eval,METH_VARARGS,"Reference tip-distance evaluation"},
 {"terminal",py_terminal,METH_VARARGS,"Position-only terminal (-1 ongoing)"},{"search",py_search,METH_VARARGS,"History-aware strict PVS"},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"ccore121",NULL,-1,methods,NULL,NULL,NULL,NULL};
PyMODINIT_FUNC PyInit_ccore121(void) { return PyModule_Create(&module); }
