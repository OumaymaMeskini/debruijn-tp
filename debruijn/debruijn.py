#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
import networkx as nx
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random
from itertools import combinations

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Your Name"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Your Name"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Your Name"
__email__ = "your@email.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, 'r') as file:
        while True:
            _ = file.readline().strip()
            sequence = file.readline().strip()
            if not sequence:
                break
            _ = file.readline().strip()
            _ = file.readline().strip()

            yield sequence



def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range((len(read)-kmer_size+1)):
        yield read[i:i+kmer_size]
    pass


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    
    # Lire les séquences à partir du fichier FASTQ
    for sequence in read_fastq(fastq_file):
        # Générer les k-mers pour chaque séquence
        for kmer in cut_kmer(sequence, kmer_size):
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1

    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    digraph = nx.DiGraph()
    
    for kmer, count in kmer_dict.items():
        # Préfixe: les k-1 premiers nucléotides du k-mer
        prefix = kmer[:-1]
        # Suffixe: les k-1 derniers nucléotides du k-mer
        suffix = kmer[1:]
        
        # Ajouter un arc du préfixe au suffixe avec un poids correspondant au nombre d'occurrences du k-mer
        digraph.add_edge(prefix, suffix, weight=count)
    
    return digraph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """

    for path in path_list:
        if not delete_entry_node and not delete_sink_node:
            graph.remove_nodes_from(path[1:-1])
        elif not delete_entry_node:
            graph.remove_nodes_from(path[1:])
        elif not delete_sink_node:
            graph.remove_nodes_from(path[:-1])
        
        else:
            graph.remove_nodes_from(path)
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """

    std_weight = statistics.stdev(weight_avg_list)
    if std_weight != 0 and len(weight_avg_list) >= 2:
        best_path_index = weight_avg_list.index(max(weight_avg_list))
    else:
        std_len = statistics.stdev(path_length)
        if std_len != 0 and len(path_length) >= 2:
            best_path_index = path_length.index(max(path_length))
        else:
            best_path_index = random.randint(0, len(path_list)-1)
    paths_to_remove = [path for index, path in enumerate(path_list) if index != best_path_index]
    remove_paths(graph, paths_to_remove, delete_entry_node, delete_sink_node)
    return graph



def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    all_paths = list(all_simple_paths(graph, ancestor_node, descendant_node))
    weight_paths = [path_average_weight(graph, path) for path in all_paths]
    length_paths = [len(path)-1 for path in all_paths]
    
    return select_best_path(graph, all_paths, length_paths, weight_paths)


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    #graph.successors(node)
    bubble = False
    ancestor_node = None
    descendant_node = None

    for node in graph.nodes():
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            for i, j in combinations(predecessors, 2):
                ancestor_node = nx.lowest_common_ancestor(graph, i, j)
                if ancestor_node is not None:
                    bubble = True
                    descendant_node = node
                    break  
            if bubble:
                break  

    if bubble:
        # Resolve the bubble using solve_bubble (assumed already implemented)
        graph = simplify_bubbles(solve_bubble(graph, ancestor_node, descendant_node))

    return graph



def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes:
        # Predescessors
        nodes_connected_to_start = [
            entry for entry in starting_nodes if nx.has_path(graph, entry, node)
        ]

        if len(nodes_connected_to_start) > 1:
            paths = []

            # Simple paths between node and his predescessors
            for node_i in nodes_connected_to_start:
                sub_paths = list(nx.all_simple_paths(graph, node_i, node))
                for path in sub_paths:
                    if len(path) > 1:
                        paths.append(path)

            if len(paths) > 1:
                weight_list = [path_average_weight(graph, path) for path in paths]
                length_list = [len(path) for path in paths]

                # Select best path
                graph = select_best_path(
                    graph, paths, length_list, weight_list,
                    delete_entry_node=True, delete_sink_node=False
                )

                starting_nodes = get_starting_nodes(graph)

                return solve_entry_tips(graph, starting_nodes)

    return graph
    


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes:
        # Trouver les successeurs qui mènent vers les nœuds de sortie
        nodes_connected_to_sink = [
            succ for succ in graph.successors(node)
            if any(nx.has_path(graph, succ, end) for end in ending_nodes) or succ in ending_nodes
        ]

        # Si un nœud a plus d'un successeur menant à des nœuds de sortie
        if len(nodes_connected_to_sink) > 1:
            paths = []

            # Collecter les chemins simples entre le nœud et les successeurs
            for succ in nodes_connected_to_sink:
                if succ in ending_nodes:
                    new_paths = [[node, succ]]  # Si le successeur est directement un nœud de sortie
                else:
                    new_paths = [
                        path for end in ending_nodes
                        for path in nx.all_simple_paths(graph, node, end)
                        if path[1] == succ  # S'assurer que le successeur est sur le chemin
                    ]

                for path in new_paths:
                    if len(path) > 1:  # Ignorer les chemins directs ou de longueur 1
                        paths.append(path)

            # S'il y a plusieurs chemins, on applique une simplification
            if len(paths) > 1:
                # Calcul des poids et des longueurs des chemins
                weight_list = [path_average_weight(graph, path) for path in paths]
                length_list = [len(path) for path in paths]

                # Sélectionner le meilleur chemin et supprimer les autres
                graph = select_best_path(
                    graph, paths, length_list, weight_list,
                    delete_entry_node=False, delete_sink_node=True
                )

                # Mettre à jour les nœuds de sortie après la simplification
                ending_nodes = get_sink_nodes(graph)

                # Appel récursif pour continuer la simplification
                return solve_out_tips(graph, ending_nodes)

    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = [node for node in graph.nodes() if len(list(graph.predecessors(node))) == 0]
    return starting_nodes

def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = [node for node in graph.nodes() if len(list(graph.successors(node))) == 0]
    return sink_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    
    # For each starting node
    for start_node in starting_nodes:
        # For each ending node
        for end_node in ending_nodes:
            if nx.has_path(graph, start_node, end_node):
                for path in nx.all_simple_paths(graph, start_node, end_node):
                    contig = path[0]  # first kmer
                    for node in path[1:]:
                        contig += node[-1]  # last caracter of kmer
                    contigs.append((contig, len(contig)))
    
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, "w") as file:
        for i, (contig, length) in enumerate(contigs_list):
            file.write(f">contig_{i} len={length}\n")
            file.write(textwrap.fill(contig, width=80) + "\n")

    pass


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    print("Building graph -------- in progress-------------")
    dict_kmer = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(dict_kmer)
    print("DONE")

    print("simplify_bubbles -------- in progress-------------")
    graph = simplify_bubbles(graph)
    print("DONE")

    print("Solving entry/out tips -------- in progress-------------")
    starting_nodes = get_starting_nodes(graph)
    graph = solve_entry_tips(graph, starting_nodes)

    ending_nodes = get_sink_nodes(graph)
    graph = solve_out_tips(graph, ending_nodes)
    print("DONE")

    print("Generating contigs -------- in progress-------------")
    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)
    contigs = get_contigs(graph, starting_nodes, ending_nodes)
    print("DONE")

    print("Saving contigs -------- in progress-------------")
    save_contigs(contigs, args.output_file)
    print("DONE")

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)

    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)






if __name__ == "__main__":  # pragma: no cover
    main()

    # Score = 13649 bits (7391),  Expect = 0.0
    # Identities = 7391/7391 (100%), Gaps = 0/7391 (0%)
    # Strand=Plus/Plus
