import pandas as pd
import numpy as np

# split the data according to value for column index
def test_split(index, value, dataset):
	left = dataset[dataset[index] <= value]
	right = dataset[dataset[index] > value]
	return left, right
	
# compute the gini index from a split
def gini(df, label):
	proportion = df.groupby(label)[label].sum()
	proportion = proportion / len(df)
	return 1 - (proportion ** 2).sum()
	
	
def calculate_score(group, label, min_size):

	if len(group['left'][label]) >= min_size and len(group['right'][label]) >= min_size:
		total_size = len(group['left']) + len(group['right'])
		score = gini(group['left'], label) * len(group['left']) / total_size + \
				gini(group['right'], label) * len(group['right']) / total_size
		return score
	
	elif len(group['left'][label]) >= min_size:
		return gini(group['left'], label) 
	
	elif len(group['right'][label]) >= min_size:
		return gini(group['right'], label) 
	
	else:
		return 1
	

# find best split, i.e. split for either right or left, impose
# to the other tree and alternate	
def get_split(group, label, protected, min_size, side='left'):
	
	# other side
	other_side = 'right'
	if side == 'right':
		other_side = 'left'

	features = [col for col in list(group['left'].columns) if (col != label) & (col != protected)]
	features_right = [col for col in list(group['right'].columns) if (col != label) & (col != protected)]
	assert (set(features) == set(features_right)), "right and left data needs the same features"
	
	index = None
	split_value = None
	new_group = {'left': None, 'right': None}
	score = 1
	
	for col in features:
		values_list = list(set(group[side][col]))
		
		for value in values_list:
			left, right = test_split(col, value, group[side])
			gini = calculate_score({'left': left, 'right': right}, label, min_size)
			
			if gini < score:
				score, split_value, index, new_group = gini, value, col, {'left': left, 'right': right}
	
	# split other tree
	if index is not None:
		left_other, right_other = test_split(index, split_value, group[other_side])
		new_trees = {}
		new_trees[side] = new_group
		new_trees[other_side] = {'left': left_other, 'right': right_other}

		if len(left_other) >= min_size and len(right_other) >= min_size:
			return {'tag': 0, 'index': index, 'value': split_value, 'score': score, 'pointer': other_side, 'left': new_trees['left']['left'], 'right': new_trees['left']['right']}, \
			{'tag': 0, 'index': index, 'value': split_value, 'score': score, 'pointer': other_side, 'left': new_trees['right']['left'], 'right': new_trees['right']['right']}
		elif len(left_other) >= min_size:
			return {'tag': 0, 'index': index, 'value': split_value, 'score': score, 'pointer': other_side, 'left': new_trees['left']['left'], 'right': pd.DataFrame()}, \
			{'tag': 0, 'index': index, 'value': split_value, 'score': score, 'pointer': other_side, 'left': new_trees['right']['left'], 'right': pd.DataFrame()}
		elif len(right_other) >= min_size:
			return {'tag': 0, 'index': index, 'value': split_value, 'score': score, 'pointer': other_side, 'right': new_trees['left']['right'], 'left': pd.DataFrame()}, \
			{'tag': 0, 'index': index, 'value': split_value, 'score': score, 'pointer': other_side, 'right': new_trees['right']['right'], 'left': pd.DataFrame()}
		else:
			return {'tag': 1, 'left':group['left'][[label]]}, {'tag': 1, 'left':group['right'][[label]]}
			
	else:
		return {'tag': 1, 'left':group['left'][[label]]}, {'tag': 1, 'left':group['right'][[label]]}
		
	
# define terminal nodes
def to_terminal(data_left, data_right, label, min_size):
	score = 0 
	if len(data_left) >= min_size and len(data_right) >= min_size:

		score_left = data_left[label].mean()
		score_right = data_right[label].mean()
		score = abs(score_right - score_left)
		return (score, data_left.index), (score, data_right.index)
	
	elif len(data_left) >= 0 and len(data_right) >= 0:
		return (score, data_left.index), (score, data_right.index)
	
	else:
		return (score, data_left), (score, data_right)
	
	
	
# construct tree by recursive splitting
def split(node_left, node_right, max_depth, min_size, depth, label, protected):
	
	if node_left['tag'] == 1:
		node_left['left'], node_right['left'] = node_left['right'], node_right['right'] = to_terminal(node_left['left'], node_right['left'], label, min_size)
		
		
		return
	if node_right['tag'] == 1 :
		node_left['left'], node_right['left'] = to_terminal(node_left['data'], node_right['data'], label, min_size)
		node_left['right'], node_right['right'] = to_terminal(node_left['data'], node_right['data'], label, min_size)
		return
		
	l_left, l_right = node_left['left'], node_left['right']
	r_left, r_right = node_right['left'], node_right['right']
	
	# check for max depth
	if depth >= max_depth:
		node_left['left'], node_right['left'] = to_terminal(l_left, r_left, label, min_size)
		node_left['right'], node_right['right'] = to_terminal(l_right, r_right, label, min_size)
		return
		
	if (len(l_left) < min_size or len(r_left) < min_size) and (len(l_right) < min_size or len(r_right) < min_size):
		node_left['left'], node_right['left'] = to_terminal(l_left, r_left, label, min_size)
		node_left['right'], node_right['right'] = to_terminal(l_right, r_right, label, min_size)
		return
		
	# process left children
	if len(l_left) < min_size or len(r_left) < min_size:
		node_left['left'], node_right['left'] = to_terminal(l_left, r_left, label, min_size)
	else:
		side = node_left['pointer']
		group = {'left': l_left, 'right': r_left}
		node_left['left'], node_right['left'] = get_split(group, label, protected, min_size, side=side)
		split(node_left['left'], node_right['left'], max_depth, min_size, depth+1, label, protected)
	
	# process right child
	if len(l_right) < min_size or len(r_right) < min_size:
		node_left['right'], node_right['right'] = to_terminal(l_right, r_right, label, min_size)
	else:
		side = node_left['pointer']
		group = {'left': l_right, 'right': r_right}
		node_left['right'], node_right['right'] = get_split(group, label, protected, min_size, side=side)
		split(node_left['right'], node_right['right'], max_depth, min_size, depth+1, label, protected)
		
# Build a decision tree
def build_tree(train_left, train_right, max_depth, min_size, label, protected):
	group = {'left': train_left, 'right': train_right}
	root_left, root_right = get_split(group, label, protected, min_size, side='left')
	split(root_left, root_right, max_depth, min_size, 1, label, protected)
	return root_left, root_right
	
def get_scores(node, depth, min_depth):

	if isinstance(node['left'], dict) and isinstance(node['right'], dict):
		left_scores = get_scores(node['left'], depth + 1, min_depth)
		right_scores = get_scores(node['right'], depth + 1, min_depth)
		return left_scores + right_scores
		
	elif isinstance(node['left'], dict):
		left_scores = get_scores(node['left'], depth + 1, min_depth)
		if node['right'][0] == 1:
			return len(node['right'][1]) + left_scores
		else:
			return left_scores
	elif isinstance(node['right'], dict):
		right_scores = get_scores(node['right'], depth + 1, min_depth)
		if node['left'][0] == 1:
			return len(node['left'][1]) + right_scores
		else:
			return right_scores
				
	else:
		if node['left'][0] == 1:
			return len(node['left'][1])
		else:
			return 0
		
def get_group(node, smax):
	
	if isinstance(node['left'], dict) and isinstance(node['right'], dict):
		
		left_scores = get_group(node['left'], smax)
		right_scores = get_group(node['right'], smax)
		
		return left_scores + right_scores
		
	elif isinstance(node['left'], dict):
		left_scores = get_group(node['left'], smax)
		if node['right'][0] >= smax:
			return left_scores + [node['right'][1]]
		else:
			return left_scores
	
	elif isinstance(node['right'], dict):
		right_scores = get_group(node['right'], smax)
		if node['left'][0] >= smax:
			return right_scores + [node['left'][1]]
		else:
			return right_scores
	else:
		if node['left'][0] >= smax or node['right'][0] >= smax:
			if len(node['left'][1]) > 0:
				if len(node['left'][1]) > 1:
					print(node)
				return [node['left'][1]]
			else:
				return []
		else:
			return []
