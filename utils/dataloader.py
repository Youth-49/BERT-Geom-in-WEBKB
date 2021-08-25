def load_data(dataset_dir, uni_name):
    """
    Load data as (node_ids, texts, labels) tuple of lists.

    Args:
        dataset_dir: String of path to the root directory of 
            pre-processed datasets.
        uni_name: String of the university name to load.
    Returns:
        A tuple of lists where (node_ids, texts, labels).
        You may need to convert labels into a tensor using stacking.
    """
    node_ids = []
    texts = []
    labels = []
    with open(dataset_dir + '/' + uni_name + '.tsv', 'r') as data_file:
        for data_line in data_file:
            node_id, text, label = data_line.split('\t')
            label = int(label)
            node_ids.append(node_id)
            texts.append(text)
            labels.append(label)
    return node_ids, texts, labels



if __name__ == '__main__':
    uni_lt = ['cornell', 'texas', 'wisconsin']
    for uni_name in uni_lt:
        node_ids, inputs, labels, sorted_data = load_data('../datasets', uni_name)
        print(node_ids[:3], inputs[:0], labels[:3])