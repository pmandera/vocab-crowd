#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random

import numpy as np
import pandas as pd


from .util import flatten, se

sessions_file = 'sessions.csv'
profiles_file = 'profiles.csv'
ua_file = 'user-agents.csv'
geo_file = 'geolocation.csv'
ld_file = 'lexical-decision.csv'


class VocabTest(object):
    def __init__(self, sessions, profiles, ld, ua, geo):
        self.sessions = sessions
        self.profiles = profiles
        self.ld = ld
        self.ua = ua
        self.geo = geo

    def spelling_sh_reliability(self, column, query=None):
        """Compute split-half reliability for words (splitting randomly on
        experimental sessions."""
        exp_ids = list(self.sessions['exp_id'])
        random.shuffle(exp_ids)
        n = len(exp_ids)
        id1 = exp_ids[:(n/2)]
        id2 = exp_ids[(n/2):]

        if not query:
            ld = self.ld
        else:
            ld = self.ld.query(query)

        ld1 = ld[ld['exp_id'].isin(id1)][['spelling', column]]
        ld2 = ld[ld['exp_id'].isin(id2)][['spelling', column]]

        means1 = ld1.groupby('spelling').mean()[column]
        means2 = ld2.groupby('spelling').mean()[column]

        return self.split_half_rel(means1, means2)

    def sample_sessions(self, n):
        """Return a VocabTest with a sample of sessions."""
        exp_ids = list(self.sessions['exp_id'])
        random.shuffle(exp_ids)
        use_exp_ids = exp_ids[:n]
        return self.subset_by_session(use_exp_ids)

    def spelling_stats(self,
                       columns=['rt', 'rt_zscore', 'accuracy'],
                       stats=[np.mean, np.std, np.count_nonzero, se],
                       query=None,
                       with_correct=True, with_error=False):
        """Compute statistics per spelling in lexical decision."""
        if not query:
            ld = self.ld
        else:
            ld = self.ld.query(query)

        all_trials = ld.groupby('spelling')[columns].aggregate(stats)

        if with_correct:
            correct_trials = ld.query('accuracy == True').groupby(
                'spelling')[columns].aggregate(stats)
            all_trials = all_trials.join(correct_trials, rsuffix='_correct')

        if with_error:
            correct_trials = ld.query('accuracy == False').groupby(
                'spelling')[columns].aggregate(stats)
            all_trials = all_trials.join(correct_trials, rsuffix='_error')

        return all_trials

    def split_half_rel(self, split1, split2):
        """Compute reliability with Spearman-Brown correction based on two
        series."""
        r = split1.corr(split2)
        return (2 * r)/(1 + r)

    def query_by_ua(self, query):
        """Return new VocabTest including only user_agents and corresponding
        data that match a query."""

        if self.ua is None:
            return None

        ua_ids = self.ua.query(query)['ua_id']
        exp_ids = self.sessions[self.sessions['ua_id'].isin(
            ua_ids)]['exp_id']
        return self.subset_by_session(exp_ids)

    def query_by_profile(self, query):
        """Return new VocabTest including only profiles and corresponding
        data that match a query (see pandas.DataFrame.query)."""
        profile_ids = self.profiles.query(query)['profile_id']
        return self.subset_by_profile(profile_ids)

    def subset_by_profile(self, profile_ids):
        """Return new VocabTest including only profiles and corresponding
        data with specified profile_ids."""
        profiles = self.profiles[self.profiles['profile_id'].isin(profile_ids)]
        sessions = self.sessions[self.sessions['profile_id'].isin(
            profiles['profile_id'])]

        if self.ua is not None:
            ua = self.ua[self.ua['ua_id'].isin(profiles['ua_id'])]
        else:
            ua = None

        if self.geo is not None:
            geo = self.geo[self.geo['exp_id'].isin(sessions['exp_id'])]
        else:
            geo = None

        ld = self.ld[self.ld['exp_id'].isin(sessions['exp_id'])]

        return VocabTest(sessions, profiles, ld, ua, geo)

    def query_by_ld(self, query):
        """Return new VocabTest including only trials matching query. In other
        tables only data corresponding to the experimental sessions with at
        least one trial will be preserved."""
        ld = self.ld.query(query)
        exp_ids = set(ld['exp_id'].unique())
        return self.subset_by_session(exp_ids, ld=ld)

    def query_by_session(self, query):
        """Return new VocabTest including only sessions matching query."""
        exp_ids = self.sessions.query(query)['exp_id']
        return self.subset_by_session(exp_ids)

    def subset_by_session(self, exp_ids, ld=None):
        """Return new VocabTest including only sessions and corresponding
        data with specified exp_ids."""
        sessions = self.sessions[self.sessions['exp_id'].isin(exp_ids)]
        profiles = self.profiles[self.profiles['profile_id'].isin(
            sessions['profile_id'])]

        if self.ua is not None:
            ua = self.ua[self.ua['ua_id'].isin(profiles['ua_id'])]
        else:
            ua = None

        if self.geo is not None:
            geo = self.geo[self.geo['exp_id'].isin(exp_ids)]
        else:
            geo = None

        if ld is None:
            ld = self.ld[self.ld['exp_id'].isin(exp_ids)]

        return VocabTest(sessions, profiles, ld, ua, geo)

    def subsets_by_profile_column(self, column):
        """Iterate all subsets of data based on values in profile column."""
        values = VocabTest.iter_unique(self.profiles[column])
        for value in values:
            if pd.isnull(value):
                continue

            if isinstance(value, str):
                query = '%s == "%s"' % (column, value)
            else:
                query = '%s == %s' % (column, value)

            subset = self.query_by_profile(query)
            yield (value, subset)

    def subsets_by_ua_column(self, column):
        """Iterate all subsets of data based on values in profile column."""

        if self.ua is None:
            return None

        values = VocabTest.iter_unique(self.ua[column])

        for value in values:
            if pd.isnull(value):
                continue

            if isinstance(value, str):
                query = '%s == "%s"' % (column, value)
            else:
                query = '%s == %s' % (column, value)

            subset = self.query_by_ua(query)
            yield (value, subset)

    @staticmethod
    def subsample_vocabtests(subsetting_f, sample=10000):
        """Subsample vocabtests based on a generator to contain equal number of
        sessions. min_n specifies what is the minimum base size to be included.
        """
        for value, subset in subsetting_f():
            yield (value, subset.sample_sessions(sample))

    def report_sizes(self):
        print('Profiles shape:', self.profiles.shape)
        print('Sessions shape:', self.sessions.shape)
        print('Ld shape:', self.ld.shape)
        print('Ld N/W counts:', self.ld.groupby(['lexicality']).size())
        print('Ld N/W correct/incorrect counts:', self.ld.groupby(
            ['lexicality', 'accuracy']).size())


    def conservative_subset(self,
                            max_profile_session=3,
                            min_trial_order=10,
                            min_rt=0,
                            max_rt=8000,
                            rt_adjbox=1,
                            min_score=0.0,
                            verbose=False):
        """Return a conservative subset of the data."""
        # this is necessary for the queries to embed
        # _local_ vars

        if max_profile_session is not None:
            q = 'profile_id_session <= %s' % max_profile_session
            self = self.query_by_session(q)

            if verbose:
                print('\n========')
                print('After max_profile_session filter:')
                self.report_sizes()

        if min_trial_order is not None:
            q = 'trial_order >= %s' % min_trial_order
            self = self.query_by_ld(q)

            if verbose:
                print('\n========')
                print('After min_trial_order filter:')
                self.report_sizes()

        if min_rt is not None:
            q = 'rt >= %s' % min_rt
            self = self.query_by_ld(q)

            if verbose:
                print('\n========')
                print('After min_rt filter:')
                self.report_sizes()

        if max_rt is not None:
            q = 'rt <= %s' % max_rt
            self = self.query_by_ld(q)

            if verbose:
                print('\n========')
                print('After max_rt filter:')
                self.report_sizes()

        if rt_adjbox is not None:
            q = 'rt_adjbox == %s' % rt_adjbox
            self = self.query_by_ld(q)

            if verbose:
                print('\n========')
                print('After rt_adjbox filter:')
                self.report_sizes()

        if min_score is not None:
            q = 'score >= %s' % min_score
            self = self.query_by_session(q)

            if verbose:
                print('\n========')
                print('After min_score filter:')
                self.report_sizes()

        return self

    def agg_ld(self, grouping=[], ld_grouping=[], var='rt',
               agg_f={'nobsa': lambda x: x.shape[0], 'var': np.mean}):
        exp_profiles = self.sessions.set_index('profile_id')[['exp_id']]
        profile_inf = self.profiles.set_index('profile_id')[grouping]
        exp_inf = exp_profiles.join(profile_inf).set_index('exp_id')[grouping]

        ld_cols = ['exp_id', var] + ld_grouping
        ld = self.ld[ld_cols].set_index('exp_id')

        full_grouping = grouping + ld_grouping
        ld_val_grouped = ld.join(exp_inf).groupby(full_grouping)[var]

        return ld_val_grouped.aggregate(agg_f).reset_index()

    def stats(self, stats={}, extras={}, flat=False, verbose=False):
        """Calculate statistics in values of a dictionary."""
        print(stats)
        result = {}
        for stat, f in list(stats.items()):

            print(stat)

            if verbose:
                print(stat, f)

            result[stat] = f(self, extras)

        if flat:
            return flatten(result)
        else:
            return result

    def add_profile_age_groups(self, age_grouping='decades'):
        """Add information about age groups to profiles.

        Age groupings: decades | education
        """
        p = self.profiles
        p['age_group'] = np.nan
        if age_grouping == 'decades':
            p.loc[(p['age'] >= 0) & (p['age'] < 10), 'age_group'] = '0 - 9'
            p.loc[(p['age'] >= 10) & (p['age'] < 20), 'age_group'] = '10 - 19'
            p.loc[(p['age'] >= 20) & (p['age'] < 30), 'age_group'] = '20 - 29'
            p.loc[(p['age'] >= 30) & (p['age'] < 40), 'age_group'] = '30 - 39'
            p.loc[(p['age'] >= 40) & (p['age'] < 50), 'age_group'] = '40 - 49'
            p.loc[(p['age'] >= 50) & (p['age'] < 60), 'age_group'] = '50 - 59'
            p.loc[(p['age'] >= 60), 'age_group'] = '>= 60'
        if age_grouping == 'education':
            p.loc[(p['age'] >= 0) & (p['age'] < 10),
                  'age_group'] = '0 - 9'
            p.loc[(p['age'] >= 10) & (p['age'] < 18),
                  'age_group'] = '10 - 17'
            p.loc[(p['age'] >= 18) & (p['age'] < 24),
                  'age_group'] = '18 - 23'
            p.loc[(p['age'] >= 24) & (p['age'] < 30),
                  'age_group'] = '24 - 29'
            p.loc[(p['age'] >= 30) & (p['age'] < 40),
                  'age_group'] = '30 - 39'
            p.loc[(p['age'] >= 40) & (p['age'] < 50),
                  'age_group'] = '40 - 49'
            p.loc[(p['age'] >= 50) & (p['age'] < 60),
                  'age_group'] = '50 - 59'
            p.loc[(p['age'] >= 60), 'age_group'] = '>= 60'

        self.profiles = p

    def add_profile_score_bins(self):
        self.add_profile_scores()
        self.profiles['score_bin'] = np.floor(self.profiles['score'] * 10)

    def add_profile_scores(self):
        """Add information about average score to each profile."""
        profile_scores = self.sessions.groupby('profile_id')['score'].mean()
        profiles_indexed = self.profiles.set_index('profile_id')
        self.profiles = profiles_indexed.join(profile_scores).reset_index()

    def save_data(self, out_dir):
        if os.path.exists(out_dir):
            raise Exception('Output directory (%s) already exist!' % out_dir)
        else:
            os.makedirs(out_dir)

        self.save_to_file(self.sessions, out_dir, sessions_file)
        self.save_to_file(self.profiles, out_dir, profiles_file)

        if self.ua is not None:
            self.save_to_file(self.ua, out_dir, ua_file)

        if self.geo is not None:
            self.save_to_file(self.geo, out_dir, geo_file)

        self.save_to_file(self.ld, out_dir, ld_file)

    def profile_session_counts(self, column):
        """Return a dictionary with number of sessions per value in column."""
        sessions = self.sessions.set_index('profile_id')
        profiles = self.profiles.set_index('profile_id')[[column]]
        return sessions.join(profiles)[column].value_counts().to_dict()

    def subset_by_profile_session_counts(self, column, min_n):
        """Subset dataset to include only profiles and associated data where we
        have at least min_n sessions for a value in a column."""
        nsessions = self.profile_session_counts(column)
        included = [k for k, v in nsessions.items() if v >= min_n]
        query = '%s in %s' % (column, included)
        return self.query_by_profile(query)

    @staticmethod
    def save_to_file(data, out_dir, fname):
        data.to_csv(os.path.join(out_dir, fname), sep='\t', index=False)

    @classmethod
    def from_dir(cls, data_dir):

        print(data_dir)

        if not os.path.exists(data_dir):
            raise Exception('Input directory (%s) does not exist!' % data_dir)

        sessions = cls.load_sessions(data_dir)
        profiles = cls.load_profiles(data_dir)
        ua = cls.load_ua(data_dir)
        geo = cls.load_geo(data_dir)
        ld = cls.load_ld(data_dir)

        return cls(sessions, profiles, ld, ua, geo)

    @staticmethod
    def load_file(data_dir, fname):
        if not os.path.exists(data_dir):
            raise Exception('Input directory (%s) does not exist!' % data_dir)
        return pd.io.parsers.read_csv(os.path.join(data_dir, fname), sep='\t')

    @classmethod
    def load_sessions(cls, data_dir):
        return cls.load_file(data_dir, sessions_file)

    @classmethod
    def load_profiles(cls, data_dir):
        return cls.load_file(data_dir, profiles_file)

    @classmethod
    def load_ld(cls, data_dir):
        return pd.io.parsers.read_csv(os.path.join(data_dir, ld_file),
                                      sep='\t',
                                      usecols=['trial_id', 'exp_id',
                                               'trial_order',
                                               # 'stim_id',
                                               'spelling', 'lexicality',
                                               'rt',
                                               'accuracy', 'response',
                                               'rt_adjbox',
                                               'rt_zscore'],
                                      dtype={
                                          'accuracy': np.int8,
                                          'trial_order': np.uint32}
                                      )

    @classmethod
    def load_ua(cls, data_dir):
        path = os.path.join(data_dir, ua_file)

        if os.path.isfile(path):
            return cls.load_file(data_dir, ua_file)
        else:
            return None

    @classmethod
    def load_geo(cls, data_dir):
        path = os.path.join(data_dir, geo_file)

        if os.path.isfile(path):
            return cls.load_file(data_dir, geo_file)
        else:
            return None

    @staticmethod
    def iter_unique(series):
        """Returns an iterator with unique values of a series."""
        return series.unique().tolist()
