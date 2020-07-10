//#include <stdio.h>
#include "PerfectMatching.h"

extern "C" {

    /**
        Integer that represents infinity (Blossom V algorithm).

        @return Integer representing infinity.
    */
    int infty() {
        return PM_INFTY;
    }

    /**
        Minimum weight perfect matching.

        @param n_nodes Number of nodes in graph.
        @param mates Empty array (length n_nodes) to be filled with matches, i.e. mates[i] = j means node i matches j.
        @param n_edges Number of edges in graph.
        @param nodes_a Node id array (length n_edges), i.e. nodes_a[k] is joined by the kth edge.
        @param nodes_b Node id array (length n_edges), i.e. nodes_b[k] is joined by the kth edge.
        @param weights Weight array (length n_edges), i.e. weights[k] is the weight of the kth edge.
    */
    void mwpm(int n_nodes, int mates[], int n_edges, int nodes_a[], int nodes_b[], int weights[]) {
        // default options (non-verbose)
        struct PerfectMatching::Options options;
        options.verbose = false;
        // create PerfectMatching
        PerfectMatching *pm = new PerfectMatching(n_nodes, n_edges);
        // add edges to PerfectMatching
        for(int i=0; i < n_edges; ++i) {
            pm->AddEdge(nodes_a[i], nodes_b[i], weights[i]);
        }
        // set options
        pm->options = options;
        // perform mwpm
        pm->Solve();
        // update mates with matches
        for (int i=0; i < n_nodes; i++) {
            mates[i] = pm->GetMatch(i);
        }
        // release PerfectMatching
        delete pm;
    }

}