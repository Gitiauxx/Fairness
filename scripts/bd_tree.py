import pandas as pd
import numpy as np

# split the data according to value for column index
def test_split(index, value, dataset):
	left = dataset[dataset[index] <= value]
	right = dataset[dataset[index] > value]
	
	return left, right
	
# compute the average score for each nodes and each tree and return the 
# max difference
def calculate_score(groups, label, min_size):
	scores = np.zeros((2, len(groups)))
	
	for i, g in enumerate(groups):
		if len(g['left'][label]) >= min_size and len(g['right'][label]) >= min_size:
			scores[i, 0] = g['left'][label].mean()
			scores[i, 1] = g['right'][label].mean()
			
		else:
			return -1, None
	
	gap = np.absolute(scores[0, :] - scores[1, :])	
	max_gap = -1
	pointer = None
	
	for i in np.arange(gap.shape[0]):
		if scores[0, i] > 0.75 or scores[0, i] < 0.25:
			if scores[1, i] > 0.75 or scores[1, i] < 0.25:
				if gap[i] > max_gap: 
					max_gap = gap[i]
					pointer = i
		
	return max_gap, pointer

# find best split, i.e. split that maximize the difference
# between data_left and data_right	
def get_split(data_left, data_right, label, protected, min_size):

	features = [col for col in list(data_left.columns) if (col != label) & (col != protected)]
	features_right = [col for col in list(data_right.columns) if (col != label) & (col != protected)]
	assert (set(features) == set(features_right)), "right and left data needs the same features"
	
	index = None
	split_value = None
	pointer = None
	group_left = {'left': None, 'right': None}
	group_right = {'left': None, 'right': None}
	
	score = -0.01 
	for col in features:
		values_list = list(set(pd.concat([data_right[col], 
						data_left[col]])))
		vmax = max(values_list)
		vmin = min(values_list)
		
		for value in values_list:
			if (value != vmax) :
				d_left_left, d_left_right = test_split(col, value, data_left)
				d_right_left, d_right_right = test_split(col, value, data_right)
			
				d_left = {'left': d_left_left, 'right': d_left_right}
				d_right = {'left': d_right_left, 'right': d_right_right}
			
				score_current, position = calculate_score([d_left, d_right], label, min_size)
				
				if score_current >= score:
					score = score_current
					split_value = value
					
					if position == 0:
						pointer = 'left'
					else:
						pointer = 'right'
					
					index = col
					group_left = d_left
					group_right = d_right

	if index is not None:
		return {'index': index, 'value': split_value, 'score': score, 'pointer': pointer, 'left': group_left['left'], 'right':group_left['right']}, \
			{'index': index, 'value': split_value, 'score': score, 'pointer': pointer, 'left': group_right['left'], 'right':group_right['right']}
	else:
		return data_left.index, data_right.index
		#return None, None
	
# define terminal nodes
def to_terminal(data_left, data_right):
	return data_left.index, data_right.index
	
# construct tree by recursive splitting
def split(node_left, node_right, max_depth, min_size, depth, label, protected):
	
	if not(isinstance(node_left, dict)) :
		return
	if not( isinstance(node_right, dict)):
		return
		
	l_left, l_right = node_left['left'], node_left['right']
	r_left, r_right = node_right['left'], node_right['right']
	
	# check for max depth
	if depth >= max_depth:
		node_left['left'], node_right['left'] = to_terminal(l_left, r_left)
		node_left['right'], node_right['right'] = to_terminal(l_right, r_right)
		return
	
	# process left children
	if len(l_left) < min_size or len(r_left) < min_size:
		node_left['left'], node_right['left'] = to_terminal(l_left, r_left)
	else:
		
		node_left['left'], node_right['left'] = get_split(l_left, r_left, label, protected, min_size)
		split(node_left['left'], node_right['left'], max_depth, min_size, depth+1, label, protected)
	
	# process right child
	if len(l_right) < min_size or len(r_right) < min_size:
		node_left['right'], node_right['right'] = to_terminal(l_right, r_right)
	else:
		node_left['right'], node_right['right'] = get_split(l_right, r_right, label, protected, min_size)
		split(node_left['right'], node_right['right'], max_depth, min_size, depth+1, label, protected)
		
# Build a decision tree
def build_tree(train_left, train_right, max_depth, min_size, label, protected):
	root_left, root_right = get_split(train_left, train_right, label, protected, min_size)
	split(root_left, root_right, max_depth, min_size, 1, label, protected)
	return root_left, root_right
	
def get_scores(node, depth, min_depth):

	if isinstance(node['left'], dict) and isinstance(node['right'], dict):
		left_scores = get_scores(node['left'], depth + 1, min_depth)
		right_scores = get_scores(node['right'], depth + 1, min_depth)
		return left_scores + right_scores
		
	elif isinstance(node['left'], dict):
		left_scores = get_scores(node['left'], depth + 1, min_depth)
		if depth >= min_depth and node['pointer'] == 'right':
			return [node['score']] + left_scores
		else:
			return left_scores
	elif isinstance(node['right'], dict):
		right_scores = get_scores(node['right'], depth + 1, min_depth)
		if depth >= min_depth and node['pointer'] == 'left':
			return [node['score']] + right_scores
		else:
			return right_scores
				
	else:
		if depth >= min_depth:
			return [node['score']]
		else:
			return []
		
def predict_score(node, row):
	
	if row[node['index']] <= node['value']:
		if isinstance(node['left'], dict):
			return predict_score(node['left'], row)
		else:
			return node['score']
	else:
		if isinstance(node['right'], dict):
			return predict_score(node['right'], row)
		else:
			return node['score']

def get_group(node):
	
	if isinstance(node['left'], dict) and isinstance(node['right'], dict):
		left_scores = get_group(node['left'])
		right_scores = get_group(node['right'])
		return left_scores.append(right_scores)
		
	elif isinstance(node['left'], dict):
		left_scores = get_group(node['left'])
		if node['pointer'] == 'right':
			return left_scores.append(node['right'])
		else:
			return left_scores
	
	elif isinstance(node['right'], dict):
		right_scores = get_group(node['right'])
		if node['pointer'] == 'left':
			return right_scores.append(node['left'])	
		else:
			return right_scores
	else:
		if node['pointer'] == 'left':
			return node['left']
		else:
			return node['right']
