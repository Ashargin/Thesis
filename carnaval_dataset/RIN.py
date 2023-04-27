#Data structure for Recurrent Interaction Networks (RINs)
#cf. "Mining for recurrent long-range interactions in RNA structures reveals embedded hierarchies in network families", NAR, 2018
#https://doi.org/10.1093/nar/gky197
#Antoine Soule <antoine.soule@polytechnique.edu> 2020

#external libraries
import json #to hash data of edges (dictionnaries)

class RIN:
    #A RIN consists in a graph and a list of "occurences" which are the RNA graphs the RIN has been found in
    graph=None #the canonical graph of the RIN
    ID=-1 #the ID of the RIN in the CaRNAval database

    #An occurence is an observation of the RIN in an RNA graph
    # as such in consists in a key corresponding to the RNA graph plus a list of nodes expliciting where the RIN has been found in the RNA graph
    nb_occurrences=0
    d_occurrences={}
    #d_occurrences is a double dictionnary :
    #lvl1 : key,d_occ (key is the (PDB_ID, chain, mode) of the RNA the RIN has been observed in)
    #lvl2 : key_occ,map (key_occ is the sorted list of node_IDs, map is a bijection of node_IDs in the canonical graph to node_IDs in the RNA)

    #The graph used to represent the RIN correspond to one of the occurences, we call this occurence the representing_occurrence
    #   Indeed, even if all occurences are isomorphique, metadata (typically the nucleotides the nodes correspond to) may vary from an occurence to the other
    representing_occurrence=None #(g_name,g_occurrence_key) Used when merging collections of RINs

    #Fields used to manipulate the RINs / collections of RINs
    order=0 #number of nodes
    size=0 #number of edges
    #In order to navigate collections more efficiently, we use two keys to divide the collections
    #the first key is made from the order and the size of the RIN to create a partial order on the RINs
    primary_key=""
    #the second key is a projection of the edge labels
    secondary_key=""

    #"canonical" is used to distinguish "temporary RINs" (potentially duplicated) created during the search from the actual RINs obtained after merging the search results
    canonical=False

    #an helping tool when comparing RINs
    d_edges_by_category={}


    #Used to store data on SSEs in the various occurences of the RINs after the first call to get_SSEs_distrib
    SSEs_distrib=None

    def __init__(self,graph,d_edges_by_category,g_name,h_name,d_generic_nodes_to_nodes_in_g,d_generic_nodes_to_nodes_in_h):
        #note : g is assumed to be the representing_occurrence
        self.graph=graph
        self.order=self.graph.order()
        self.size=self.graph.size()
        self.primary_key='o'+str(self.order).zfill(6)+'s'+str(self.size).zfill(6) #zfill so primary keys are ordered

        self.canonical=False
        self.ID=-1

        self.d_edges_by_category=d_edges_by_category
        self.secondary_key=json.dumps(d_edges_by_category, sort_keys=True)

        self.d_occurrences={}
        g_occurrence_key=json.dumps(sorted(d_generic_nodes_to_nodes_in_g.values()))
        self.representing_occurrence=(g_name,g_occurrence_key)
        self.d_occurrences[g_name]={g_occurrence_key:d_generic_nodes_to_nodes_in_g}
        h_occurrence_key=json.dumps(sorted(d_generic_nodes_to_nodes_in_h.values()))
        self.d_occurrences[h_name]={h_occurrence_key:d_generic_nodes_to_nodes_in_h}
        self.nb_occurrences=2

    def make_canonical(self,ID):
        self.canonical=True
        self.ID=ID

    def get_keys(self):
        return (self.primary_key,self.secondary_key)

    def create_occurrence_graphs(self,data):
        #as this operation is to be performed once and since copy is only used in it
        #we prefer to exceptionnaly place the import inside the function
        import copy
        l_occurrence_graphs=[]
        for key,d_occ in self.d_occurrences.items():
            PDB_ID,chain,suffix=key
            for key_occ,map in d_occ.items():
                g=copy.deepcopy(data[(PDB_ID,chain)])
                nodes_to_keep = list(map.values())
                nodes_to_remove = list(set(g.nodes())-set(nodes_to_keep))
                g.remove_nodes_from(nodes_to_remove)
                l_occurrence_graphs.append(((PDB_ID,chain),suffix,key_occ,g))
        return l_occurrence_graphs

    def get_SSEs_distrib(self,data=None):
        if self.SSEs_distrib == None:
            if data!=None:
                self.SSEs_distrib=self.get_SSEs(data)
        return self.SSEs_distrib


    def get_SSEs(self,data):
        #it's a total overkill to use create_occurrence_graphs just to get the distributions because of copy
        distrib={}
        for key,d_occ in self.d_occurrences.items():
            PDB_ID,chain,suffix=key
            RNA_graph=data[(PDB_ID,chain)]

            for key_occ,map in d_occ.items():
                d_sse_in_occ={}
                n_to_cover_1=[]
                d_pid={}
                for node in map.values():
                    d_pid[node]=RNA_graph.nodes[node]["part_id"]
                    if len(d_pid[node]) <2:
                        d_sse_in_occ[d_pid[node][0]]=True
                    else:
                        n_to_cover_1.append(node)

                n_to_cover_2=[]
                for node in n_to_cover_1:
                    covered=False
                    for part_id in d_pid[node]:
                        if part_id in d_sse_in_occ.keys():
                            covered=True
                    if not covered:
                        n_to_cover_2.append(node)
                while n_to_cover_2 != []:
                    d_sse={}
                    for node in n_to_cover_2:
                        for sse in d_pid[node]:
                            tmp=d_sse.get(sse,[])
                            tmp.append(node)
                            d_sse[sse]=tmp
                    candidate_sse=''
                    candidate_count=0
                    for sse,l_nodes in d_sse.items():
                        if len(l_nodes) > candidate_count:
                            candidate_sse=sse
                            candidate_count=len(l_nodes)
                    d_sse_in_occ[candidate_sse]=True
                    for node in d_sse[candidate_sse]:
                        n_to_cover_2.remove(node)


                distrib[len(d_sse_in_occ)]=distrib.get(len(d_sse_in_occ),0)+1
        return distrib
