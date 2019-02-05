# [Python] Vocabulary crowdsourcing project

This is the code supporting analysis of the data from the vocabulary
crowdsourcing project. Allows to easily subset and aggregate the data.

Compatible with:

* English Crowdsourcing Project dataset [dataset](http://)
    [www](http://vocabulary.ugent.be)

(other datasets in preparation)

# Installation

```
git clone git@github.com:pmandera/vocab-crowd.git
cd vocab-crowd
python3 -m pip install -r requirements.txt --user
python3 setup.py install
```

# Usage

To load the dataset:

```
from vocabtest.vocabtest import VocabTest

vt = VocabTest.from_dir('./english-vocabtest-20180919-native.lang.en/')
```

To compute lexical statistics based on a subset of data:

```
print(vt.profiles.head())

# use only data from female participants younger than 25
vt_female = vt.query_by_profile('gender == "Female" and age < 25')

print(vt_female.profiles.head())

# calculate average statistics for all words but skip trials 0-9
# and use only those not filtered out by the
# adjusted boxplot method (Hubert & Vandervieren, 2008)
w_stats = vt_female.spelling_stats(
  query='lexicality == "W" and trial_order > 9 and rt_adjbox == True')

print(w_stats.head())
```

# Authors

The tool was developed by [Paweł Mandera](http://www.pawelmandera.com/).

If you are using this code for scientific purposes please cite:

Mandera, P., Keuleers, E., & Brysbasert, M. (submitted). Recognition times for
62 thousand English words: Data from the English Crowdsourcing Project.

# License

The project is licensed under the Apache License 2.0.
