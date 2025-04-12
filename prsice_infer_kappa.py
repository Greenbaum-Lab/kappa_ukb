import sys
import json
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

def generate_pair_comparisons(pgs, sample_ids, phenotypes, first_subset_size=0, unique_samples=0):
	"""
	Generates comparisons among subsets of data assuming that the subset is comprised of two equal-sized same-sex groups
	:param first_subset_size: Can be used to specify the size of the first same-sex group if the groups are not of equal size
	"""
	split_n = first_subset_size if first_subset_size != 0 else int(len(pgs) / 2)
	pgs_subset_f = pgs[:split_n]
	phenotypes_subset_f = phenotypes[:split_n]
	pgs_subset_m = pgs[split_n:]
	phenotypes_subset_m = phenotypes[split_n:]
	if unique_samples == 0:
		first_group = np.array(np.triu_indices(k=1, n=split_n)).T
		second_n = len(pgs) - split_n
		second_group = np.array(np.triu_indices(k=1, n=second_n)).T + len(pgs) - second_n
		all_pairs = np.concatenate([first_group, second_group])
	elif unique_samples == 1:
		all_pairs = np.arange(len(pgs)).reshape(int(len(pgs)/2), 2)
	pairs = all_pairs[~np.any(np.isnan(phenotypes[all_pairs]), axis=1)]
	dpgs = np.diff(pgs[pairs], axis=1)
	dphenotype = np.diff(phenotypes[pairs], axis=1)

	# Optional: Run regression separately if needed
	# pgs_regress_f = perform_regression(pgs_subset_f, phenotypes_subset_f)
	# pgs_regress_m = perform_regression(pgs_subset_m, phenotypes_subset_m)
	return dpgs[:,0], np.sign(dpgs * dphenotype)[:,0], pairs

def infer_rbar(dpgs, phenotypic_difference, r_bar_range, r_bar_preset=[], bootstrap_p=0, kappa_bins=51):
	"""
	Infers r_bar based on the fit of the proportion of matches per kappa bin to the theoretical probability of match
	and computes bootstrapped confidence intervals for P.

	:param dpgs: ndarray of dPGS values for all comparisons
	:param phenotypic_difference: ndarray of the signs of the phenotypic difference between all comparisons
	:param r_bar_range: range of values to infer r_bar values if no presets are provided
	:param r_bar_preset: preset values for r_bar inference and kappa computation
	:param bootstrap_p: Number of bootstrap samples for confidence intervals; 0 disables bootstrapping
	:param kappa_bins: number of bins for kappa histogram
	:return: Tuple containing the optimal r_bar, output data for plotting, and optional bootstrap results
	"""
	dpgs_norm = np.abs(dpgs / np.std(dpgs))
	bins = np.linspace(0, 1, kappa_bins)
	expected = norm.cdf(bins / (1 - bins))
	r_bar_diff = []
	r_bar_data = []
	r_bar_values = r_bar_preset if len(r_bar_preset) > 0 else np.arange(*r_bar_range, 0.01)

	def bootstrap_ci(values, weights, confidence_level=0.95, n_resamples=1000, subset_size=100000000):
		"""
		Manually perform weighted bootstrapping with limited subset size.

		:param values: Array of data values.
		:param weights: Array of weights corresponding to values.
		:param confidence_level: Desired confidence level (e.g., 0.95).
		:param n_resamples: Number of bootstrap iterations.
		:param subset_size: Maximum number of samples per bootstrap iteration.
		:return: Lower and upper confidence intervals.
		"""

		n = len(values)
		subset_size = min(n, subset_size)

		resampled_means = []
		for _ in range(n_resamples):
			# Subsample a fixed number of points with replacement
			subset_indices = np.random.choice(n, size=subset_size, replace=True)
			resampled_values = values[subset_indices]
			resampled_weights = weights[subset_indices]
			# Calculate the weighted mean for the resampled subset
			resampled_mean = np.average(resampled_values, weights=resampled_weights)
			resampled_means.append(resampled_mean)
		# Compute confidence intervals
		alpha = (1 - confidence_level) / 2
		lower = np.percentile(resampled_means, alpha * 100)
		upper = np.percentile(resampled_means, (1 - alpha) * 100)
		return lower, upper

	for r_bar in r_bar_values:
		# Compute kappa and binning
		kappa = r_bar * dpgs_norm / (r_bar * dpgs_norm + np.power(1 - np.power(r_bar, 2), 0.5))
		print(np.mean(kappa))
		kappa_p = norm.cdf(kappa / (1 - kappa))
		bin_indices = np.digitize(kappa, bins) - 1
		valid_bins = bin_indices < (len(bins) - 1)
		bin_indices = bin_indices[valid_bins]

		# Calculate non-zero and positive counts
		n_non_zero = np.bincount(bin_indices, weights=(phenotypic_difference[valid_bins] != 0).astype(int), minlength=len(bins) - 1)
		n_positive = np.bincount(bin_indices, weights=(phenotypic_difference[valid_bins] > 0).astype(int), minlength=len(bins) - 1)

		# Proportions and bootstrapped confidence intervals
		P = np.where(n_non_zero > 0, n_positive / n_non_zero, 1)
		bootstrap_results = []
		if bootstrap_p == 1:
			for i in range(len(P)):
				if n_non_zero[i] > 0:
					try:
						ci = bootstrap_ci(
							values=(phenotypic_difference[valid_bins][bin_indices == i] > 0).astype(int),
							weights=(phenotypic_difference[valid_bins][bin_indices == i] != 0).astype(int)
						)
						print(ci)
						bootstrap_results.append((P[i], ci))
					except Exception:
						bootstrap_results.append((P[i], None))
				else:
					bootstrap_results.append((P[i], None))
		else:
			bootstrap_results = [(p, None) for p in P]

		# Data for fitting
		data_points = np.column_stack((bins[:-1], P, n_non_zero))
		diff = np.sum(np.power(data_points[:, 1] - expected[:-1], 2) * data_points[:, 2])
		print([r_bar, diff])
		r_bar_diff.append([r_bar, diff])
		r_bar_data.append({
			'data_points': data_points,
			'bootstrap_results': bootstrap_results,
			'kappa_hist': np.array([bins[:-1], np.histogram(kappa, bins=bins, density=True)[0]]).T,
			'p_hist': np.array([np.linspace(0, 1, 21)[:-1], np.histogram(kappa_p, bins=np.linspace(0, 1, 21), density=True)[0]]).T,
		})

	# Select optimal r_bar
	min_diff_i = np.argsort(np.array(r_bar_diff)[:, 1])[0]
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
	dpgs, phenotypic_difference, pair_indices = generate_pair_comparisons(pgs, sample_ids, phenotype_values, params['first_subset_size'], params['unique_samples'])
	r_bar, kappa_data, dpgs, dpgs_norm = infer_rbar(dpgs, phenotypic_difference, r_bar_range, r_bar_preset, params['bootstrap'])
	
	return r_bar, kappa_data

def save_results(params, results):
	prefix = params['prefix']
	r_bar, kappa_data = results
	np.savetxt(f'{prefix}_kappa_P.csv', kappa_data['data_points'], delimiter=',', header='kappa,P,n', comments='')
	if params['bootstrap'] == 1:
		with open(f'{prefix}_kappa_P_bootstrap.json', 'w') as file:
			file.write(json.dumps(kappa_data['bootstrap_results']))
	np.savetxt(f'{prefix}_kappa_hist.csv', kappa_data['kappa_hist'], delimiter=',', header='kappa,density', comments='')
	np.savetxt(f'{prefix}_p_hist.csv', kappa_data['p_hist'], delimiter=',', header='P,density', comments='')
	print(f'Best fit r_bar value: {r_bar}')

def setup_parser():
	parser = argparse.ArgumentParser(description='Infer kappa from PRSice data')
	parser.add_argument('--dataset-directory', required=True, help='The directory containing the PRSice results (/prsice) and lists of sample IDs and phenotypes (/subsets)')
	parser.add_argument('--subset', required=True, help='The label of the subset of sample IDs to use')
	parser.add_argument('--gwas', required=True, help='The label of the GWAS that appears in the PRSice result filename')
	parser.add_argument('--phenotype', required=True, help='The label of the phenotype, that appears in the PRSice result filename, and the file containing the phenotypes')
	parser.add_argument('--prefix', required=False, default='ukb_result', help='The prefix of the output files')
	parser.add_argument('--first-subset-size', type=int, required=False, default=0, help='By default we compute all comparisons between samples within the first and second half of the samples separately. This manually sets the point at which to split the subset')
	parser.add_argument('--unique-samples', type=int, required=False, default=0, help='Whether we should use all comparisons or use the order of samples in the subset file (1=use pairs from file)')
	parser.add_argument('--r-bar-range', required=False, default='0.1,0.5', help='A comma-delimited range of \bar{r} values to use for inference')
	parser.add_argument('--r-bar-preset', required=False, default='', help='Instead of inferring \bar{r}, use a pre-defined value')
	parser.add_argument('--bootstrap', required=False, type=int, default=0, help='Whether to run bootstrapping on the P values to generate CIs (1=run bootstrapping)')
	return parser

def main():
	parser = setup_parser()
	params = vars(parser.parse_args())
	results = run(params)
	save_results(params, results)

if __name__ == '__main__':
	main()

# python3 prsice_infer_kappa.py --prefix=eur_height --dataset-directory=../ukbb/data --subset=eur --gwas=EUR-subset --phenotype=height --r-bar-preset=0.43
