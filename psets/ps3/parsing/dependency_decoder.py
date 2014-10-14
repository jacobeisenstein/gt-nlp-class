import sys
import numpy as np


class DependencyDecoder():
    '''
    Dependency decoder class
    '''
    def __init__(self):
        self.verbose = False

    def parse_marginals_nonproj(self, scores):
        '''
        Compute marginals and the log-partition function using the matrix-tree theorem
        '''
        nr, nc = np.shape(scores)
        if nr != nc:
            raise ValueError("scores must be a squared matrix with nw+1 rows")
            return []

        nw = nr - 1;

        s = np.matrix(scores)
        lap = np.matrix(np.zeros((nw+1, nw+1)))
        for m in range(1, nw + 1):
            d = 0.0
            for h in range(0, nw + 1):
                if m != h:
                    d += np.exp(s[h,m])
                    lap[h,m] = -np.exp(s[h,m])
            lap[m,m] = d
        r = lap[0,1:]
        minor = lap[1:,1:]

        #logZ = np.linalg.slogdet(minor)[1]
        logZ = np.log(np.linalg.det(minor))
        invmin = np.linalg.inv(minor)
        marginals = np.zeros((nw+1, nw+1))
        for m in range(1, nw + 1):
            marginals[0,m] = np.exp(s[0,m]) * invmin[m-1,m-1]
            for h in range(1, nw + 1):
                if m != h:
                    marginals[h,m] = np.exp(s[h,m]) * (invmin[m-1,m-1] - invmin[m-1,h-1])

        return marginals, logZ

        


    def parse_nonproj(self, scores):
        '''
        Parse using Chu-Liu-Edmonds algorithm
        '''
        nr, nc = np.shape(scores)
        if nr != nc:
            raise ValueError("scores must be a squared matrix with nw+1 rows")
            return []

        nw = nr - 1;

        curr_nodes = np.ones(nw+1, int)
        reps = []
        oldI = -np.ones((nw+1, nw+1), int)
        oldO = -np.ones((nw+1, nw+1), int)
        for i in range(0, nw+1): 
            reps.append({i : 0})
            for j in range(0, nw+1):
                oldI[i,j] = i
                oldO[i,j] = j
                if i==j or j==0:
                    continue

        if self.verbose:
            print "Starting C-L-E...\n"

        scores_copy = scores.copy()
        final_edges = self.chu_liu_edmonds(scores_copy, curr_nodes, oldI, oldO, {}, reps)
        heads = np.zeros(nw+1, int)
        heads[0] = -1
        for key in final_edges.keys():
            ch = key
            pr = final_edges[key]
            heads[ch] = pr

        #ok = False
        #for m in range(np.size(heads)):
        #    if heads[m] == 0:
        #        ok = True
        #        break
        #if not ok and np.size(heads) < 5:
        #    heads[1000000] = -1

        return heads


    def chu_liu_edmonds(self, scores, curr_nodes, oldI, oldO, final_edges, reps):
        '''
        Chu-Liu-Edmonds algorithm
        '''

        # need to construct for each node list of nodes they represent (here only!)
        nw = np.size(curr_nodes) - 1;




        # create best graph
        par = -np.ones(nw+1, int)
        for m in range(1, nw+1):
            # only interested in current nodes
            if 0 == curr_nodes[m]: 
                continue
            max_score = scores[0,m]
            par[m] = 0
            for h in range(nw+1):
                if m == h:
                    continue
                if 0 == curr_nodes[h]:
                    continue
                if scores[h,m] > max_score:
                    max_score = scores[h,m]
                    par[m] = h

        if self.verbose:
            print "After init\n"
            for m in range(0, nw+1):
                if 0 < curr_nodes[m]:
                    print "{0}|{1} ".format(par[m],m)
            print "\n"


        # find a cycle
        cycles = []
        added = np.zeros(nw+1, int)
        for m in range(0, nw+1):
            if np.size(cycles) > 0:
                break
            if added[m] or 0 == curr_nodes[m]:
                continue
            added[m] = 1
            cycle = {m : 0}
            l = m
            while True:
                if par[l] == -1:
                    added[l] = 1
                    break
                if par[l] in cycle:
                    cycle = {}
                    lorg = par[l]
                    cycle[lorg] = par[lorg]
                    added[lorg] = 1
                    l1 = par[lorg]
                    while l1 != lorg:
                        cycle[l1] = par[l1]
                        added[l1] = True
                        l1 = par[l1]
                    cycles.append(cycle)
                    break
                cycle[l] = 0
                l = par[l]
                if added[l] and (l not in cycle):
                    break
                added[l] = 1

        # get all edges and return them
        if np.size(cycles) == 0:
            for m in range(0, nw+1):
                if 0 == curr_nodes[m]:
                    continue
                if par[m] != -1:
                    pr = oldI[par[m], m]
                    ch = oldO[par[m], m]
                    final_edges[ch] = pr
                else:
                    final_edges[0] = -1
            return final_edges

        max_cyc = 0
        wh_cyc = 0
        for cycle in cycles:
            if np.size(cycle.keys()) > max_cyc:
                max_cyc = np.size(cycle.keys())
                wh_cyc = cycle

        cycle = wh_cyc
        cyc_nodes = cycle.keys()
        rep = cyc_nodes[0]
        
        if self.verbose:
            print "Found Cycle\n"
            for node in cyc_nodes:
                print "{0} ".format(node)
            print "\n"

        cyc_weight = 0.0
        for node in cyc_nodes:
            cyc_weight += scores[par[node], node] 

        for i in range(0, nw+1):
            if 0 == curr_nodes[i] or (i in cycle):
                continue

            max1 = -np.inf 
            wh1 = -1
            max2 = -np.inf
            wh2 = -1

            for j1 in cyc_nodes:
                if scores[j1, i] > max1:
                    max1 = scores[j1, i]
                    wh1 = j1

                # cycle weight + new edge - removal of old
                scr = cyc_weight + scores[i, j1] - scores[par[j1], j1]
                if scr > max2:
                    max2 = scr
                    wh2 = j1

            scores[rep, i] = max1
            oldI[rep, i] = oldI[wh1, i]
            oldO[rep, i] = oldO[wh1, i]
            scores[i, rep] = max2
            oldO[i, rep] = oldO[i, wh2]
            oldI[i, rep] = oldI[i, wh2]

        rep_cons = []
        for i in range(0, np.size(cyc_nodes)):
            rep_con = {}
            keys = sorted(reps[int(cyc_nodes[i])].keys())
            if self.verbose:
                print "{0}: ".format(cyc_nodes[i])
            for key in keys:
                rep_con[key] = 0
                if self.verbose:
                   print "{0} ".format(key)
            rep_cons.append(rep_con)
            if self.verbose:
                print "\n"

        # don't consider not representative nodes
        # these nodes have been folded
        for node in cyc_nodes[1:]:
            curr_nodes[node] = 0
            for key in reps[int(node)]:
                reps[int(rep)][key] = 0

        self.chu_liu_edmonds(scores, curr_nodes, oldI, oldO, final_edges, reps)

        # check each node in cycle, if one of its representatives
        # is a key in the final_edges, it is the one.
        if self.verbose:
            print final_edges
        wh = -1;
        found = False
        for i in range(0, np.size(rep_cons)): 
            if found:
                break
            for key in rep_cons[i]: 
                if found:
                    break
                if key in final_edges:
                    wh = cyc_nodes[i]
                    found = True
        l = par[wh]
        while l != wh:
            ch = oldO[par[l]][l]
            pr = oldI[par[l]][l]
            final_edges[ch] = pr
            l = par[l]

        return final_edges

