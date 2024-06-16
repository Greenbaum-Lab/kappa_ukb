import sys
import io
import math
import argparse
import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import linregress, norm

def load_phenotypes(dataset_directory, subset, phenotype):
	"""
	Loads a file containing the phenotypes of the subset, one phenotype value per line, following the order of the sample list file. Phenotypes are extracted from the UKB phenotype table according to the Data-Field of the specific phenotype
	"""
	file_path = f'{dataset_directory}/subsets/phenotypes_{subset}_{phenotype}.txt'
	return np.array(pd.read_csv(file_path, names=['phenotype'], delimiter=' ', skip_blank_lines=False)['phenotype'])

def load_sample_list(dataset_directory, subset):
	"""
	Loads a list of UKB sample IDs (needs to correspond with sample IDs in PRSice files), the order of which follows that of the phenotype value file
	"""
	file_path = f'{dataset_directory}/subsets/{subset}.txt'
	return np.array(pd.read_csv(file_path, names=['sample_id'], delimiter=' ')['sample_id'])

def load_pgs_scores(dataset_directory, gwas, phenotype, sample_ids):
	"""
	Loads polygenic scores (PGS) for a given phenotype and GWAS from a PRSice output file listing the PGS value per sample
	"""
	file_path = f'{dataset_directory}/prsice/{phenotype}/{gwas}.all_score'
	sample_scores = pd.read_csv(file_path, delimiter=' ', header=0, names=['FID', 'sample_id', 'PGS'], usecols=['sample_id', 'PGS'], index_col='sample_id')
	return np.array(sample_scores.loc[sample_ids]['PGS'])

def perform_regression(pgs_subset, phenotypes_subset):
	"""
	Performs linear regression on non-NaN phenotype and genotype data to infer the r^2 value for the subset
	"""
	valid_indices = ~np.isnan(phenotypes_subset)
	return linregress(pgs_subset[valid_indices], phenotypes_subset[valid_indices])

def generate_pair_comparisons(pgs, phenotypes, first_subset_size=0):
	"""
	Generates comparisons among subsets of data assuming that the subset is comprised of two equal-sized same-sex groups
	:param first_subset_size: Can be used to specify the size of the first same-sex group if the groups are not of equal size
	"""
	split_n = first_subset_size if first_subset_size != 0 else int(len(pgs) / 2)
	pgs_subset_f = pgs[:split_n]
	phenotypes_subset_f = phenotypes[:split_n]
	pgs_subset_m = pgs[split_n:]
	phenotypes_subset_m = phenotypes[split_n:]

	first_group = np.array(np.triu_indices(k=1, n=split_n)).T
	second_n = len(pgs) - split_n
	second_group = np.array(np.triu_indices(k=1, n=second_n)).T + len(pgs) - second_n
	all_pairs = np.concatenate([first_group, second_group])

	pairs = all_pairs[~np.any(np.isnan(phenotypes[all_pairs]), axis=1)]
	dpgs = np.diff(pgs[pairs], axis=1)
	dphenotype = np.diff(phenotypes[pairs], axis=1)

	# Optional: Run regression separately if needed
	# pgs_regress_f = perform_regression(pgs_subset_f, phenotypes_subset_f)
	# pgs_regress_m = perform_regression(pgs_subset_m, phenotypes_subset_m)

	return dpgs[:,0], np.sign(dpgs * dphenotype)[:,0], pairs

def infer_rbar(dpgs, phenotypic_difference, r_bar_range, r_bar_preset=[], kappa_bins=51):
	"""
	Infers r_bar based on the fit of the proportion of matches per kappa bin to the theoretical probability of match and returns the kappa values for the optimal r_bar value
	
	:param dpgs: ndarray of dPGS values for all comparisons
	:param phenotypic_difference: ndarray of the signs of the phenotypic difference between all comparisons
	:param r_bar_range: range of values to infer r_bar values if no presets are provided
	:param r_bar_preset: preset values for r_bar inference and kappa computation (for instance, can be used if r_bar has been inferred in a previous step)
	:param kappa_bins: number of bins for kappa histogram
	:return: Tuple containing the optimal r_bar and output data for plotting
	"""
	dpgs_norm = np.abs(dpgs / np.std(dpgs))
	bins = np.linspace(0, 1, kappa_bins)
	expected = norm.cdf(bins / (1 - bins))
	r_bar_diff = []
	r_bar_data = []
	r_bar_values = r_bar_preset if len(r_bar_preset) > 0 else np.arange(*r_bar_range, 0.01)
	for r_bar in r_bar_values:
		kappa = r_bar * dpgs_norm / (r_bar * dpgs_norm + np.power(1 - np.power(r_bar, 2), 0.5))
		kappa_p = norm.cdf(kappa / (1 - kappa))
		bin_indices = np.digitize(kappa, bins) - 1
		valid_bins = bin_indices < (len(bins) - 1)
		bin_indices = bin_indices[valid_bins]
		n_non_zero = np.bincount(bin_indices, weights=(phenotypic_difference[valid_bins] != 0).astype(int), minlength=len(bins) - 1)
		n_positive = np.bincount(bin_indices, weights=(phenotypic_difference[valid_bins] > 0).astype(int), minlength=len(bins) - 1)
		P = np.where(n_non_zero > 0, n_positive / n_non_zero, 1)
		data_points = np.column_stack((bins[:-1], P, n_non_zero))
		diff = np.sum(np.power(data_points[:,1] - expected[:-1], 2) * data_points[:,2])
		print([r_bar, diff])
		r_bar_diff.append([r_bar, diff])
		r_bar_data.append({
			'data_points': data_points,
			'kappa_hist': np.array([bins[:-1], np.histogram(kappa, bins=bins, density=True)[0]]).T,
			'p_hist': np.array([np.linspace(0, 1, 21)[:-1], np.histogram(kappa_p, bins=np.linspace(0, 1, 21), density=True)[0]]).T,
		})
	min_diff_i = np.argsort(np.array(r_bar_diff)[:,1])[0]
	return r_bar_diff[min_diff_i][0], r_bar_data[min_diff_i], dpgs, dpgs_norm

def run(params):
	try:
		r_bar_range = np.array(params['r_bar_range'].split(','), dtype=float)
		r_bar_preset = [] if params['r_bar_preset'] == '' else np.array(params['r_bar_preset'].split(','), dtype=float)
	except ValueError as e:
		raise ValueError(f"Error parsing range or preset values: {e}")

	phenotype_values = load_phenotypes(params['dataset_directory'], params['subset'], params['phenotype'])
	sample_ids = load_sample_list(params['dataset_directory'], params['subset'])
	pgs = load_pgs_scores(params['dataset_directory'], params['gwas'], params['phenotype'], sample_ids)
	dpgs, phenotypic_difference, pair_indices = generate_pair_comparisons(pgs, phenotype_values, params['first_subset_size'])
	r_bar, kappa_data, dpgs, dpgs_norm = infer_rbar(dpgs, phenotypic_difference, r_bar_range, r_bar_preset)
	
	return r_bar, kappa_data

def save_results(prefix, results):
	r_bar, kappa_data = results
	np.savetxt(f'{prefix}_kappa_P.csv', kappa_data['data_points'], delimiter=',', header='kappa,P,n', comments='')
	np.savetxt(f'{prefix}_kappa_hist.csv', kappa_data['kappa_hist'], delimiter=',', header='kappa,density', comments='')
	np.savetxt(f'{prefix}_p_hist.csv', kappa_data['p_hist'], delimiter=',', header='P,density', comments='')
	print(f'Best fit r_bar value: {r_bar}')

def setup_parser():
	parser = argparse.ArgumentParser(description='Infer kappa from PRSice data')
	parser.add_argument('--dataset-directory', required=True)
	parser.add_argument('--subset', required=True)
	parser.add_argument('--gwas', required=True)
	parser.add_argument('--phenotype', required=True)
	parser.add_argument('--prefix', required=False, default='ukb_result')
	parser.add_argument('--first-subset-size', type=int, required=False, default=0)
	parser.add_argument('--r-bar-range', required=False, default='0.1,0.5')
	parser.add_argument('--r-bar-preset', required=False, default='')
	return parser

def main():
	parser = setup_parser()
	params = vars(parser.parse_args())
	results = run(params)
	save_results(params['prefix'], results)

if __name__ == '__main__':
	main()
