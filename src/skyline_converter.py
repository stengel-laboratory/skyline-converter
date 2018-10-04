#!/usr/bin/env python3.6

import sys
import pandas as pd
import numpy as np
# note the more general import statsmodels.stats.api as sms leads to warnings which I want to avoid
import statsmodels.sandbox.stats.multicomp as sm
import statsmodels.stats.weightstats as smw
import argparse
import math
from typing import List,Dict
import best3


# TODO: properly evaluate diff. label output (review paper)
# TODO: add some sort of quality assessment from skyline (Isotope Dot Product is already exported, maybe use cutoff)
# TODO: add container (and support) for technical replicates
# TODO: export intensity of first isotope and use that instead of MS1 area (like xtract) if ms1 area == 0
# TODO: maybe add some kde (kernel density estimator) for different parameters, like log2ratios, ms1 areas, pvals, etc..

desc = """Kai Kammer - 2018-02-06. 
Script to convert xTract input files to Skyline input files, linearizing the crosslinks in the process. Also supports 
MS1 quantitation of Skyline output files. Last Update: 04.05.2018 
"""

parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-ix', '--input_xtract', action="store", dest="input_xt", default=None,
                    help="xTract input (for conversion from xTract input to Skyline input)"
                         " or Skyline input file name for quantification (i.e. the converted xTract file). Required!")
parser.add_argument('-is', '--input_skyline', action="store", dest="input_sk", default=None,
                    help="Skyline output file name. Required for quantification")
parser.add_argument('-er', '--experiment_ref', action="store", dest="exp_ref", default=None,
                    help="Name of the reference experiment (as in the Skyline output) for quantification")
parser.add_argument('-b', '--bayesian', action="store_true", dest="bayesian", default=False,
                    help="Calculate a bayesian estimator for the efffect size. Time intensive, at least 15s more"
                         " per crosslink. Requires the BEST python package")
parser.add_argument('-bp', '--bayesian_plot', action="store_true", dest="bayesian_plot", default=False,
                    help="Same as -b option but will also save plots of the posteriors of the mean, sd and effect size.")
parser.add_argument('-f', '--filter', action="store", dest="filter", default=None, type=str,
                    help="Optionally provide a key here (e.g. q-val) for filtering the output csv file")
parser.add_argument('-fv', '--filter_value', action="store", dest="filter_val", default="0.05", type=str,
                    help="If using a filter provide the threshold here for filtering below or above that")
parser.add_argument('-fo', '--filter_operator', action="store", dest="filter_op", default="gt", type=str,
                    help="If using a filter decide whether values below the filter value are dropped (lt)"
                         " or above (gt) or equal values (eq)."
                         " Possible options: lt (lower than), gt (greater than), eq (equal)")
parser.add_argument('-ff', '--filter_file', action="store", dest="filter_file", default="is", type=str,
                    help="Which file is to be filtered."
                         " Possible values: is (input Skyline); ix (input xTract)")
parser.add_argument('-o', '--outname', action="store", dest="outname", default='out_skyline_converter.csv',
                    help="Name for the output file")
parser.add_argument('-p', '--path_mzxml', action="store", dest="path_mzxml",
                    default='Z:/Data/mzXML/profile/',
                    help="Path to profile xml dir when converting xTract to Skyline input")
args = parser.parse_args()
seq_string_xtract = 'seq'
seq_string_original = 'seq_original'
seq_string_skyline = 'sequence'
pep_modified_seq_string = 'PeptideModifiedSequenceMonoisotopicMasses'
ms1_area_string = 'TotalAreaMs1'
replicate_string = 'BioRep'
experiment_string = 'ExpName'
uxid_string = 'uxID'
type_string = 'type'
mono_string = 'monolink'
inter_string = 'xlink'
intra_string = 'intralink'
heavy_string = 'heavy'
light_string = 'light'
hydrolyzed_string = 'hydro'
ammo_string = 'ammo'
log2ratio_string = 'log2ratio'
string_conv_dict = {'q-val': 'DetectionQValue'}


# Print progress
# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class XtractContainer(object):
    # TODO: take care of loop link as in: https://link.springer.com/article/10.1007%2Fs13361-017-1837-2#Sec2
    # BS3-loop (138.068 Da)
    # monolink: light:155
    # C8O2NH13  # Delta mass by the cross-linker (ammonium quenched, 1H is subtracted)
    weight_mono_light_nh2 = 155.0946
    # monolink: heavy:155
    # C8O2NHD12  # Delta mass for heavy xlinker
    weight_mono_heavy_nh2 = 167.1699
    # monolink: light:156
    # C8O3H12  # Delta mass by the cross-linker (hydrolized, 1H is subtracted)
    weight_mono_light_oh = 156.0786
    # monolink: heavy:156
    # C8O3D12  # Delta mass for heavy xlinker
    weight_mono_heavy_oh = 168.1540

    # H atom
    weight_h = 1.007825
    # equals mass xlinker (without leaving groups) + m(H2O) - (Mod1_light * 2 + m(lys); "m(C2O2)" - "m(H2N2)
    # monoisotopic mass!
    weight_diff_mod2_light = 25.96803

    # D atom
    weight_d = 2.014102
    # equals mass xlinker (without leaving groups) + m(H2O) - (Mod1_heavy * 2 + m(lys); "m(C2O2D12)" - "m(H14N2)
    # or exactly mod2_light + 12 * m(neutron)
    # monoisotopic mass!
    weight_diff_mod2_heavy = 38.04335

    def __init__(self, entry):
        self.pep_seq = entry[seq_string_original]
        self.xl_type, self.weight_type = self._get_type_weight(entry[type_string])
        self.pep_1, self.pep_2, self.site_1, self.site_2, self.mono_type = self._get_peps_and_link_sites()
        self.weight_mod_1 = self._get_weight_mod_1()
        self.weight_mod_2 = self._get_weight_mod_2()
        self.linear_seq = self._get_linear_seq()


    def _get_peps_and_link_sites(self):
        pep_seq_split = self.pep_seq.split('-')
        pep_1, pep_2 = "None", "None"
        site_1, site_2 = -1, -1
        mono_type = 'None'
        if self.xl_type == inter_string:
            pep_1 = pep_seq_split[0]
            pep_2 = pep_seq_split[1]
            site_1 = int(pep_seq_split[2].replace('a', ''))
            site_2 = int(pep_seq_split[3].replace('b', ''))
        elif self.xl_type == intra_string:
            pep_1 = pep_seq_split[0]
            pep_2 = "None"
            site_1 = int(pep_seq_split[1].replace('K', ''))
            site_2 = int(pep_seq_split[2].replace('K', ''))
        elif self.xl_type == mono_string:
            pep_1 = pep_seq_split[0]
            pep_2 = "None"
            site_1 = int(pep_seq_split[1].replace('K', ''))
            site_2 = -1
            if pep_seq_split[2] == "155":
                mono_type = ammo_string
            elif pep_seq_split[2] == "156":
                mono_type = hydrolyzed_string
        return pep_1, pep_2, site_1, site_2, mono_type

    def _get_type_weight(self, type_weight):
        return type_weight.split(':')

    def _get_linear_seq(self):
        lin_seq = "None"
        if self.xl_type == inter_string:
            pep1 = self.pep_1[:self.site_1] + "[+{0}]".format(self.weight_mod_1) + self.pep_1[self.site_1:]
            pep2 = self.pep_2[:self.site_2] + "[+{0}]".format(self.weight_mod_1) + self.pep_2[self.site_2:]
            x_mod = "K[+{0}]".format(self.weight_mod_2)
            lin_seq = pep1 + x_mod + pep2
        elif self.xl_type == intra_string:
            pep1 = self.pep_1[:self.site_1] + "[+{0}]".format(self.weight_mod_1)
            pep2 = self.pep_1[self.site_1:self.site_2] + "[+{0}]".format(self.weight_mod_1) + self.pep_1[self.site_2:]
            x_mod = "K[+{0}]".format(self.weight_mod_2)
            lin_seq = pep1 + x_mod + pep2
        elif self.xl_type == mono_string:
            lin_seq = self.pep_1[:self.site_1] + "[+{0}]".format(self.weight_mod_1) + self.pep_1[self.site_1:]
        return lin_seq

    def _get_weight_mod_1(self):
        weight = -1.0
        if self.xl_type == inter_string or self.xl_type == intra_string:
            if self.weight_type == light_string:
                weight = self.weight_h
            elif self.weight_type == heavy_string:
                weight = self.weight_d
        elif self.xl_type == mono_string:
            if self.weight_type == light_string:
                if self.mono_type == ammo_string:
                    weight = self.weight_mono_light_nh2
                elif self.mono_type == hydrolyzed_string:
                    weight = self.weight_mono_light_oh
            elif self.weight_type == heavy_string:
                if self.mono_type == ammo_string:
                    weight = self.weight_mono_heavy_nh2
                elif self.mono_type == hydrolyzed_string:
                    weight = self.weight_mono_heavy_oh
        return weight

    def _get_weight_mod_2(self):
        weight = -1.0
        if self.xl_type == inter_string or self.xl_type == intra_string:
            if self.weight_type == light_string:
                weight = self.weight_diff_mod2_light
            elif self.weight_type == heavy_string:
                weight = self.weight_diff_mod2_heavy
        return weight

    def __str__(self):
        return "{4} {5} container: P1: {0}, P2: {1}, S1: {2}, S2: {3}. Mono type: {6}\n" \
               "linear seq: {7}".format(self.pep_1, self.pep_2,
                                        self.site_1, self.site_2,
                                        self.weight_type, self.xl_type,
                                        self.mono_type, self.linear_seq)

    # using the linear sequence to compare identity; it contains all the xlink information
    # and allows for comparison between Xtract and SkylineContainers
    def __eq__(self, other):
        if self.linear_seq == other.linear_seq:
            return True
        return False

    def __hash__(self):
        return hash(self.linear_seq)


class SkylineContainer(XtractContainer):

    def __init__(self, entry):
        self.ms1_area = int(entry[ms1_area_string])
        self.experiment = entry[experiment_string]
        self.replicate = entry[replicate_string]
        self.uxid = None
        self.pep_seq = entry[pep_modified_seq_string]
        entry[seq_string_original] = self.pep_seq
        entry[type_string] = None
        super().__init__(entry)

    @staticmethod
    def _round_weights(weight):
        # IMPORTANT: every weight which is compared has to round to n-1 where n are the decimal places of the
        #           most inaccurate weight which is 4 right now
        return round(weight, 3)

    def _get_peps_and_link_sites(self):
        pep_seq_split = self.pep_seq.split(']')
        pep_1, pep_2 = "None", "None"
        site_1, site_2 = -1, -1
        mono_type = 'None'
        # we've got an inter- or intralink
        if len(pep_seq_split) == 4:
            # get peptides between the brackets and remove fake lysine for pep1
            pep_1 = pep_seq_split[0].split('[')[0] + pep_seq_split[1].split('[')[0][:-1]
            pep_2 = pep_seq_split[2].split('[')[0] + pep_seq_split[3].split('[')[0]
            indexes_open = list(substring_indexes('[', self.pep_seq))
            site_1 = indexes_open[0]
            site_2 = pep_seq_split[2].find('[')
        # monolink
        elif len(pep_seq_split) == 2:
            pep_1 = pep_seq_split[0].split('[')[0] + pep_seq_split[1]
            pep_2 = "None"
            start = self.pep_seq.find('[')
            end = self.pep_seq.find(']')
            site_1 = start
            site_2 = -1
            mod_weight = self._round_weights(float(self.pep_seq[start+1:end]))
            #print("WEIGHT: ", mod_weight)
            if mod_weight == self._round_weights(self.weight_mono_light_nh2) or mod_weight == self._round_weights(self.weight_mono_heavy_nh2):
                mono_type = ammo_string
            elif mod_weight == self._round_weights(self.weight_mono_light_oh) or mod_weight == self._round_weights(self.weight_mono_heavy_oh):
                mono_type = hydrolyzed_string
        return pep_1, pep_2, site_1, site_2, mono_type

    def _get_type_weight(self, type_weight=None):
        start = self.pep_seq.find('[')
        end = self.pep_seq.find(']')
        type = 'None'
        weight = 'None'
        weight_mod1 = self._round_weights(float(self.pep_seq[start+1:end]))
        rounded_weights_light = [self._round_weights(self.weight_h), self._round_weights(self.weight_mono_light_oh), self._round_weights(self.weight_mono_light_nh2)]
        rounded_weights_heavy = [self._round_weights(self.weight_d), self._round_weights(self.weight_mono_heavy_oh), self._round_weights(self.weight_mono_heavy_nh2)]
        #print(weight_mod1, rounded_weights_light)
        if weight_mod1 in rounded_weights_light:
            weight = light_string
        elif weight_mod1 in rounded_weights_heavy:
            weight = heavy_string
        pep_seq_split = self.pep_seq.split(']')
        if len(pep_seq_split) == 4:
            # make all xlinks interlinks as we can't differentiate in skyline output between intra-/interlinks
            type = inter_string
        elif len(pep_seq_split) == 2:
            type = mono_string
        return type, weight

    def set_uxid(self, uxid):
        self.uxid = uxid

    def __str__(self):
        return super().__str__()  + "\n" \
                                    "experiment: {0}; replicate: {1}; ms1 area: {2}\n" \
                                    "uxID: {3}".format(self.experiment, self.replicate, self.ms1_area, self.uxid)

    def _count_extra_skyline_chars(self):
        """ function to count the number of extra chars added by skyline
            not needed after all; will see if that changes and leave the function for now"""
        indexes_open = list(substring_indexes('[', self.pep_seq))
        indexes_closed = list(substring_indexes(']', self.pep_seq))
        cnt = 0
        for n, index_open in enumerate(indexes_open):
            start = index_open
            end = indexes_closed[n]
            cnt += end - start + 1
        return cnt


class TotalContainer(object):
    """ All containers inherit from TotalContainer
        The succession is as follows and determined by the self.child attribute and the get_child_params method:
        TotalCont->UXCont->ExpCont->ReplCont->LightHeavyCont
        MS1 areas may be split according to this hierarchy
        Skyline containers are appended in this way as well, each representing a peptide"""
    def __init__(self):
        self.skyline_container_list = []  # type: List[SkylineContainer]
        self.sub_container_dict = {}  # type: Dict{str: UXContainer}
        self.child = UXContainer
        self.ms1_sum = 0
        self.ms1_highest = 0  # highest ms1 area among all peptides; alternative to ms1 sum but not yet used
        self.name = "TotalContainer"

        self.type = self.__class__.__name__

    def __str__(self):
        return "ContainerType: {0}; ID: {1}; MS1 Area: {2}".format(self.type, self.name, self.ms1_sum)

    def _get_child_param(self, sky_cont: SkylineContainer):
        return sky_cont.uxid

    def add(self, sky_cont: SkylineContainer):
        self.skyline_container_list.append(sky_cont)
        child_param = self._get_child_param(sky_cont)
        if child_param in self.sub_container_dict:
            self.sub_container_dict[child_param].add(sky_cont)
        else:
            sub_cont = self.child(child_param)
            self.sub_container_dict[child_param] = sub_cont
            self.sub_container_dict[child_param].add(sky_cont)

    def get_ms1_dict(self):
        exp_ms1_dict = {}
        for sub_name,sub_cont in self.sub_container_dict.items():
            exp_ms1_dict[sub_name] = sub_cont.ms1_sum
        return exp_ms1_dict

    def sum_up(self):
        for sub_cont in self.sub_container_dict.values():
            sub_cont.sum_up()
        for sky_cont in self.skyline_container_list:
            self.ms1_sum += sky_cont.ms1_area
            if sky_cont.ms1_area > self.ms1_highest:
                self.ms1_highest = sky_cont.ms1_area

    def get_basic_stats(self):
        """ calculates the mean, standard deviation, coefficient of variation, standard error of the mean
            and 95% confidence interval (95% chance we find the mean within this interval)"""
        val_list = []
        for name, sub_cont in self.sub_container_dict.items():
            val_list.append(sub_cont.ms1_sum)
        mean = np.mean(val_list)
        sd = np.std(val_list)
        cov = sd/mean
        # sem = sd/math.sqrt(len(val_list))
        # ci95 = 1.96 * sem
        return mean, sd, cov


    def get_subcontainer_size(self):
        # return int and whether all sub containers' subs have the same size and if yes which, otherwise -1
        a = np.empty(len(self.sub_container_dict))
        for n, sub_con in enumerate(self.sub_container_dict.values()):
            a[n] = len(sub_con.sub_container_dict)
        size_mean = a.mean()
        size_prop = (a == size_mean).mean()
        if size_prop == 1.0:
            return int(size_mean)
        else:
            return -1



    def get_two_sample_ttest(self, name_exp, name_ref, equal_var=True):
        """ two sample t-test from statsmodels
            will take the sublists of the children
            for example if called on a UXContainer the samples will be the replicates of our two experiments
            UXContainer->ExperimentContainer->ReplicateContainer
            returns t-statistic (effect size) and p-value"""
        if name_exp in self.sub_container_dict and name_ref in self.sub_container_dict:
            sub_exp = self.sub_container_dict[name_exp]
            sub_ref = self.sub_container_dict[name_ref]
            sub_exp_child_ms1s = []
            sub_ref_child_ms1s = []
            # add values in for-loop because dict.values() yields strings and we need ints
            for val in sub_exp.get_ms1_dict().values():
                sub_exp_child_ms1s.append(val)
            for val in sub_ref.get_ms1_dict().values():
                sub_ref_child_ms1s.append(val)
            cm = smw.CompareMeans.from_data(sub_exp_child_ms1s, sub_ref_child_ms1s)

            if equal_var:
                usevar = 'pooled'
            else:
                usevar = 'unequal'
            # note that usevar='pooled' (or equal_var = true) means we are using Student's t-test
            # while false (usevar='unequal') means we're using Welch's version
            # for small small and equal sample sizes size the Student's version seems to be preferred
            # but there is a lot of discussion: https://stats.stackexchange.com/questions/305/when-conducting-a-t-test-why-would-one-prefer-to-assume-or-test-for-equal-vari
            # also "unequal variances are only a problem when group sizes are unequal": http://daniellakens.blogspot.de/2015/01/always-use-welchs-t-test-instead-of.html
            # scipy version: ss.ttest_ind(sub_ref_child_ms1s, sub_exp_child_ms1s, equal_var=True)
            # also returns the dof in position [3] which we don't need
            # tconfint_diff() allows for a confidence intervall
            return cm.ttest_ind(usevar=usevar)[:2]
        else:
            print("WARNING: Trying to ttest a non-existent key: {0}; {1}".format(name_exp, name_ref))
            # return zero effect size and high pval if the keys are missing for some reason
            return 0,1

    def get_log2ratio(self, name_exp, name_ref):
        # TODO: what to do when either of the areas is zero
        # maybe like xtract: if either are not found set to detection limit
        # which is  ~10^5 to 10^6 for the Orbitrap Fusion
        # if area_ref == 0:
        #     area_ref = 10E5
        # if area_comp == 0:
        #     area_comp = 10E-5
        if name_exp in self.sub_container_dict and name_ref in self.sub_container_dict:
            sub_exp = self.sub_container_dict[name_exp]
            sub_ref = self.sub_container_dict[name_ref]
            exp_ms1 = sub_exp.ms1_sum
            ref_ms1 = sub_ref.ms1_sum


            if ref_ms1 == 0 or exp_ms1 == 0:
                return -1
            log2ratio = math.log(exp_ms1 / ref_ms1, 2)
            # calculating a SEM and CI95 does only make sense on the ms1 area level, not on the log2ratio level
            # only viable if we'd take the mean log2ratio between replicates which in turn would invalidate the t-test
            # exp_ci_lower, exp_ci_upper = sub_exp.get_ci95()
            # ref_ci_lower, ref_ci_upper = sub_ref.get_ci95()
            # exp_sem = sub_exp.get_sem()
            # ref_sem = sub_ref.get_sem()
            # log2sem = math.log(exp_sem / ref_sem, 2)
            # sub_exp_child_ms1s = []
            # sub_ref_child_ms1s = []
            # log2list = []
            # for val in sub_exp.get_ms1_dict().values():
            #     sub_exp_child_ms1s.append(val)
            # for val in sub_ref.get_ms1_dict().values():
            #     sub_ref_child_ms1s.append(val)
            # for n in range(len(sub_exp_child_ms1s)):
            #     log2list.append(math.log(sub_exp_child_ms1s[n] / sub_ref_child_ms1s[n], 2))
            # sc = smw.DescrStatsW(log2list)
            # mean, sem = sc.mean, sc.stdm
            # lower, upper = sc.tconfint_mean()
            return log2ratio
        else:
            print("WARNING: Trying to get a log2ratio for a non-existent key: {0}; {1}".format(name_exp, name_ref))
            # just returning 0 here, which is probably not the best idea
            return 0

    def get_ci95(self):
        ms1_list = []
        for sub_cont in self.sub_container_dict.values():
            ms1_list.append(sub_cont.ms1_sum)
        sc = smw.DescrStatsW(ms1_list)
        return sc.tconfint_mean()

    def get_sem(self):
        ms1_list = []
        for sub_cont in self.sub_container_dict.values():
            ms1_list.append(sub_cont.ms1_sum)
        cm = smw.DescrStatsW(ms1_list)
        return cm.std_mean

    def get_bayesian_estimator(self, name_exp, name_ref, plot_name=None):
        sub_a = self.sub_container_dict[name_exp]
        sub_b = self.sub_container_dict[name_ref]
        sub_a_child_ms1s = []
        sub_b_child_ms1s = []
        # add values in for-loop because dict.values() yields strings and we need ints
        for val in sub_a.get_ms1_dict().values():
            sub_a_child_ms1s.append(val)
        for val in sub_b.get_ms1_dict().values():
            sub_b_child_ms1s.append(val)
        report = best3.sample_data({name_exp: sub_a_child_ms1s, name_ref: sub_b_child_ms1s}, plot_name=plot_name)
        rep_dict = report.to_dict(orient='index')
        return rep_dict['effect size']

class UXContainer(TotalContainer):
    def __init__(self, name):
        super().__init__()
        self.child = ExperimentContainer
        self.name = name

    def _get_child_param(self, sky_cont: SkylineContainer):
        return sky_cont.experiment


class ExperimentContainer(TotalContainer):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.child = ReplicateContainer


    def _get_child_param(self, sky_cont: SkylineContainer):
        return sky_cont.replicate


class ReplicateContainer(TotalContainer):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.child = LightHeavyContainer

    def _get_child_param(self, sky_cont: SkylineContainer):
        return sky_cont.weight_type

class LightHeavyContainer(TotalContainer):
    def __init__(self, name):
        super().__init__()
        self.name = name
        # last container in the hierarchy
        self.child = None

    def add(self, sky_cont):
        self.skyline_container_list.append(sky_cont)


def filter_by_key(df):
    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    filter_name = args.filter

    if filter_name in string_conv_dict:
        filter_name = string_conv_dict[filter_name]

    df = df.dropna(subset=[filter_name])
    df = df.reset_index()
    print("Shape before filtering via {0} {1} {2}: {3}.".format(filter_name, args.filter_op, args.filter_val,
                                                                df.shape))
    if isfloat(args.filter_val):
        filter_val = float(args.filter_val)
        df[filter_name] = df[filter_name].astype(float)
    else:
        filter_val = args.filter_val
    if args.filter_op == 'lt':
        df = df.drop(df[df[filter_name] < filter_val].index)
    elif args.filter_op == 'gt':
        df = df.drop(df[df[filter_name] > filter_val].index)
    elif args.filter_op == 'eq':
        df = df.drop(df[df[filter_name] == filter_val].index)
    # default is lt
    else:
        df = df.drop(df[df[filter_name] < filter_val].index)
    print("Shape after filtering via {0} {1} {2}: {3}.".format(filter_name, args.filter_op, args.filter_val,
                                                                df.shape))
    return df


def convert_xtract_to_skyline(df: pd.DataFrame):
    df = df.rename(columns={'scan': 'scan_original', seq_string_xtract: seq_string_original, 'z': 'charge'})
    # TODO: which score type is the best? Or rather none? Ask at the message board! Discussion ongoing....
    # Using PEPTIDE PROPHET SOMETHING right now as it is one of the scores describing probability of matching
    # See https://skyline.ms/wiki/home/software/BiblioSpec/page.view?name=BiblioSpec%20input%20and%20output%20file%20formats
    # df['score_type'] = 'PEPTIDE PROPHET SOMETHING'
    df['file'] = 'n/a'
    df['scan'] = 'n/a'
    df[seq_string_skyline] = 'n/a'
    total_peps = df.shape[0]
    printProgressBar(0, total_peps, prefix='Progress:', suffix='Complete', length=50)
    dict_df = df.to_dict(orient='index')
    for n in range(len(dict_df)):
        printProgressBar(n + 1, total_peps, prefix='Parsing peptide {0}/{1}:'.format(n + 1, total_peps),
                         suffix='Complete', length=50)
        link = XtractContainer(dict_df[n])
        # df.at is faster at accessing than df.loc due to being scalar only
        # http://pandas.pydata.org/pandas-docs/stable/indexing.html#fast-scalar-value-getting-and-setting
        df.at[[n], seq_string_skyline] = link.linear_seq
        split_scan = df['scan_original'].values[n]
        split_scan = split_scan.split('.')
        df.at[[n], 'file'] = "{0}{1}{2}".format(args.path_mzxml, split_scan[0], '.mzXML')
        df.at[[n], 'scan'] = "{0}".format(split_scan[1])
    return df


def analyze_skyline_out_match_uxid(df_xtract: pd.DataFrame, df_skyline: pd.DataFrame):
    if args.exp_ref is None:
        print("WARNING: No reference experiment name specified, aborting")
        exit(0)

    # dropping NaNs from the skyline output, i.e. peptides which do not have a corresponding ms1 spectrum match
    df_skyline = df_skyline.dropna(subset=[ms1_area_string])
    df_skyline = df_skyline.reset_index()
    # converting the dataframes to dicts before iterating increases performance by several orders of magnitude
    # index orientation allows access to elements via index->column->value
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_dict.html
    dict_skyline = df_skyline.to_dict(orient='index')
    dict_xtract = df_xtract.to_dict(orient='index')

    total_peps = df_skyline.shape[0]
    total_cont = TotalContainer()
    non_matched_counter = 0
    printProgressBar(0, total_peps, prefix='Progress:', suffix='Complete', length=50)
    for n in range(len(dict_skyline)):
        # use \r and end="" to go back one line and print to the same location
        # however much cooler with progress bar, therefore the next line is commented
        # print("Parsing peptide {0} out of {1} ({2:.2%})\r".format(n + 1, total_peps, (n + 1) / total_peps), end=""),
        printProgressBar(n + 1, total_peps, prefix='Parsing peptide {0}/{1}:'.format(n + 1, total_peps), suffix='Complete', length=50)
        entry_sky = dict_skyline[n]
        link_sky = SkylineContainer(entry_sky)
        cnt = 0
        #for m in range(len(df_xtract)):
        for m in range(len(dict_xtract)):
            link_xt = XtractContainer(dict_xtract[m])
            # we found a matching link
            if link_sky == link_xt:
                cnt += 1
                # set uxid of the skyline link
                link_sky.set_uxid(dict_xtract[m][uxid_string])
                total_cont.add(link_sky)
                break
        if cnt == 0:
            print("WARNING: No uxID was matched for Skyline link: {0}".format(link_sky))
            non_matched_counter += 1
    if non_matched_counter > 0: print("There were {0} peptides for which no uxID was found. They will not be quantified."
          .format(non_matched_counter))
    print("Found {0} uxIDs for quantification. Calculating statistics now"
          .format(len(total_cont.sub_container_dict)))
    return total_cont


def analyze_skyline_out_sum_up_ms1(total_cont: TotalContainer):
    total_cont.sum_up()
    raw_dict = {}
    # we need the pval list to apply corrections to them (bonf, simes-hochberg fdr)
    pval_list = []
    uxid_cont: UXContainer
    for uxid, uxid_cont in total_cont.sub_container_dict.items():
        exp_cont: ExperimentContainer
        # only evaluate if we have at least two experiments for a given uid and both experiments should have at least 2 replicates
        if len(uxid_cont.sub_container_dict) >= 2 and uxid_cont.get_subcontainer_size() >=2:
            for exp_comp_name, exp_cont in uxid_cont.sub_container_dict.items():
                if exp_comp_name != args.exp_ref:
                    # do not compare reference experiment with itself
                    # using a two-sample t-test as in xTract for the biological replicates between the experiments
                    t_stat, pval = uxid_cont.get_two_sample_ttest(exp_comp_name, args.exp_ref)
                    if args.bayesian or args.bayesian_plot:
                        if args.bayesian_plot:
                            plot = uxid
                        else:
                            plot = None
                        # bayesian_effect_stats = uxid_cont.get_bayesian_estimator(args.exp_ref, exp_comp_name, plot)
                        # raw_dict = add_to_raw_data(bayesian_effect_stats, raw_dict, "bay_effect ")
                        bayesian_effect_stats3 = uxid_cont.get_bayesian_estimator(exp_comp_name, args.exp_ref, plot)
                        raw_dict = add_to_raw_data(bayesian_effect_stats3, raw_dict, "bay_effect ")
                    pval_list.append(pval)
                    raw_dict = add_to_raw_data({'uID': uxid}, raw_dict)
                    raw_dict = add_to_raw_data({"t_stat": t_stat}, raw_dict)
                    raw_dict = add_to_raw_data({"t_pval": pval}, raw_dict)
                    # log2ratio, log2sem, log2ci_lower, log2ci_upper = uxid_cont.get_log2ratio(exp_comp_name, args.exp_ref)
                    log2ratio = uxid_cont.get_log2ratio(exp_comp_name, args.exp_ref)
                    raw_dict = add_to_raw_data({'log2ratio': log2ratio}, raw_dict)
                    # raw_dict = add_to_raw_data({'log2sem': log2sem}, raw_dict)
                    # raw_dict = add_to_raw_data({'log2ci95_lower': log2ci_lower}, raw_dict)
                    # raw_dict = add_to_raw_data({'log2ci95_upper': log2ci_upper}, raw_dict)
                    raw_dict = add_to_raw_data({'experiment information':  'exp: {0}| ref: {1}'
                                           .format(exp_comp_name, args.exp_ref)}, raw_dict)
    # correction for multiple testing as the actual alpha error increases with the number of tests
    # a_act = 1 - p(non_significant) = 1 - (1 - a)^n
    # bonferroni method is simple but conserative -> increases false negatives while decreasing false positives
    # note that bonferroni is just bonf = pval * n with n being the number of tests (i.e. len(p_val_list))
    pval_list_bonf = sm.multipletests(pval_list, method='bonf')[1]  # only interested in the corrected pvals
    # fdr is estimated by calculating q-values according to Benjamini and Hochberg
    # q_val_i = p_val_i * n/i with p_vals sorted ascending, so lower p-values will have a higher increase
    # however, theyy need to be corrected if the increase is higher than a succeeding p-value
    # min(q_val) = min(bonf_val) and max(q_val) = max(p_val)
    pval_list_fdr_bh = sm.multipletests(pval_list, method='fdr_bh')[1]
    raw_dict["bonf"] = pval_list_bonf
    raw_dict["fdr"] = pval_list_fdr_bh
    len_raw = check_raw_dict(raw_dict)
    dropped_uids = len(total_cont.sub_container_dict) - len_raw
    if dropped_uids > 0:
        print("{0} uxIDs were dropped due to missing ms1 data for an experiment.".format(dropped_uids))
    df = pd.DataFrame(raw_dict)
    cols = df.columns.tolist() #Type: List
    cols = sorted(cols, reverse=True)
    df = df[cols]
    df = df.sort_values(by='fdr')
    return df


def check_raw_dict(raw_dict):
    len_vals = -1
    uniform = True
    for key, vals in raw_dict.items():
        if len(vals) == len_vals or len_vals == -1:
            len_vals = len(vals)
        else:
            uniform = False
            break
    if uniform:
        return len_vals
    else:
        print("Raw dict has non-uniform length. Will exit now")
        for key, vals in raw_dict.items():
            print("{0} has length {1}".format(key, len(vals)))
        exit(0)




def add_to_raw_data(dict_to_add, raw_dict, prefix=None):
    for key, value in dict_to_add.items():
        if prefix is not None:
            raw_dict[prefix + key] = raw_dict.get(prefix + key, []) + [value]
        else:
            raw_dict[key] = raw_dict.get(key, []) + [value]
    return raw_dict


def substring_indexes(substring, string):
    """
    Generate indices of where substring begins in string
    In the end not used but still a nice function
    >>> list(substring_indexes('me', "The cat says meow, meow"))
    [13, 19]
    """
    last_found = -1  # Begin at -1 so the next position to search from is 0
    while True:
        # Find next index of substring, by starting after its last known position
        last_found = string.find(substring, last_found + 1)
        if last_found == -1:
            break  # All occurrences have been found
        yield last_found


def main():
    assert sys.version_info >= (3, 6), "Python version should be at least 3.6"
    df_xtract = None
    df_skyline = None
    df = None
    ff = None
    if args.input_xt:
        df_xtract = pd.read_csv(args.input_xt, sep=None, engine='python')
        if seq_string_original in df_xtract or seq_string_xtract in df_xtract:
            print("xTract input read successfully")
        else:
            print("WARNING: No compatible xTract input file detected; aborting")
            exit(0)
    if args.input_sk:
        df_skyline = pd.read_csv(args.input_sk, sep=None, engine='python')
        if pep_modified_seq_string in df_skyline:
            print("Skyline input read successfully")
        else:
            print("WARNING: No compatible skyline input file detected; aborting")
            exit(0)
    if args.filter is not None:
        if args.filter_file is 'ix':
            df_xtract = filter_by_key(df_xtract)
        elif args.filter_file is 'is':
            df_skyline = filter_by_key(df_skyline)
    if df_xtract is not None and df_skyline is not None:
        tot_cont = analyze_skyline_out_match_uxid(df_xtract, df_skyline)
        df = analyze_skyline_out_sum_up_ms1(tot_cont)
        # limiting significant digits to 6
        ff = '%.6g'
    elif df_xtract is not None:
        df = convert_xtract_to_skyline(df_xtract)
    if df is not None:
        df.to_csv(args.outname, index=False, line_terminator='\n', sep="\t", float_format=ff)
        print("Successfully exported to {0} with shape {1}.".format(args.outname, df.shape))

if __name__ == "__main__":
    main()
