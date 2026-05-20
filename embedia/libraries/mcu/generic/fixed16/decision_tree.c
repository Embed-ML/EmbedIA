/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 * GitHub: https://github.com/Embed-ML/EmbedIA
 */
#include "decision_tree.h"

/**
 * Internal helper to traverse the decision tree.
 */
static node_index_t dtree_traverse(decision_tree_classifier_layer_t tree, compute_t *instance)
{
    node_index_t id = 0;

    while(!DT_IS_LEAF(tree.nodes[id])) {

        if(instance[tree.nodes[id].feature_id] <= tree.nodes[id].threshold) {
            id = id + 1;
        } else {
            id = id + tree.nodes[id].right_offset;
        }

    }

    return id;
}



void decision_tree_classifier_layer(decision_tree_classifier_layer_t tree, data1d_t input, data1d_t* output)
{
    output->length = tree.n_classes;
    output->data = (compute_t*)swap_alloc(sizeof(compute_t) * tree.n_classes);

    compute_t* instance = input.data;

    node_index_t id = dtree_traverse(tree, instance);

    class_id_t predicted_class = tree.nodes[id].value;

    // Initialize all classes to 0, set predicted class to 1.0
    for (int i = 0; i < tree.n_classes; i++) {
        output->data[i] = (i == predicted_class) ? COMPUTE_ONE : COMPUTE_ZERO;
    }
}