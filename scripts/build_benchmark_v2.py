"""Build benchmark v2 from v1 by deduplicating, dropping dark items, and fixing Cat 1+5.

v1: data/benchmark/benchmark.jsonl   (62 items)
v2: data/benchmark/benchmark_v2.jsonl (46 items)

Changes:
- Drop 10 duplicate items (one kept per anchor group)
- Drop 6 dark items where both models score <=20% on both conditions (no tax signal)
- Rewrite perturbed_prompt for 7 items to remove information leakage / meta-prompts
- All other fields (test_a, test_b, ref_solution_a/b, interpretation_a/b) untouched
- task_ids preserved (no renumbering) so traceability to v1 is direct
"""

import json
from pathlib import Path

V1 = Path("data/benchmark/benchmark.jsonl")
V2 = Path("data/benchmark/benchmark_v2.jsonl")

# === Items dropped because they share an anchor with a kept item =============
# Format: dropped_id -> kept_id (for documentation only)
DROP = {
    "AMBI/018": "AMBI/005",  # max_sum_increasing_subseq
    "AMBI/028": "AMBI/011",  # count_Occurrence (also Cat 1 leak — dropped covers fix)
    "AMBI/029": "AMBI/011",  # count_Occurrence
    "AMBI/045": "AMBI/003",  # maximize_elements (also Cat 1 leak)
    "AMBI/046": "AMBI/006",  # convert_list_dictionary (also Cat 1 leak)
    "AMBI/027": "AMBI/037",  # frequent value (both dark; kept the scopal-typed one)
    "AMBI/041": "AMBI/040",  # unique ID per `a`
    "AMBI/044": "AMBI/043",  # reverse + concat (043 has cleaner tax signal)
    "AMBI/050": "AMBI/051",  # overlapping rows (050 dark; 051 keeps after fix)
    "AMBI/053": "AMBI/055",  # z-score (both dark)
}

# === Dark items: both models score <=20% on baseline AND pass_either ==========
# These cannot contribute to Ambiguity Tax (the metric is undefined under
# floor-effect conditions). Coverage in DS-1000 elliptical and MBPP syntactic
# will be replenished later with HumanEval-sourced items.
DARK = {
    "AMBI/012": "mbpp/syntactic — task too hard for SOTA models",
    "AMBI/032": "mbpp/syntactic — canonical test cases internally inconsistent",
    "AMBI/037": "ds1000/scopal — brittle output schema; survived dedup but still dark",
    "AMBI/055": "ds1000/elliptical — row z-score output format too specific; survived dedup but still dark",
    "AMBI/059": "ds1000/elliptical — string-format output is brittle",
    "AMBI/061": "ds1000/elliptical — clean prompt's target itself unclear",
}

# === Rewritten perturbed_prompts for remaining Cat 1+5 items ==================
# Each entry: task_id -> new perturbed_prompt (full string).
FIXES = {}

# AMBI/006 — drop the (l1, l2, l3) signature leak; use *lists.
FIXES["AMBI/006"] = (
    "def convert_list_dictionary(*lists):\n"
    '    """Convert more than one list to a nested dictionary.\n'
    '    """'
)

# AMBI/010 — revert docstring to the clean prompt's vague wording.
FIXES["AMBI/010"] = (
    "def index_multiplication(test_tup1, test_tup2):\n"
    '    """\n'
    "    Perform index-wise multiplication of tuple elements in the given two tuples.\n"
    '    """'
)

# AMBI/033 — replace the inadvertent simplification with a real binding ambiguity.
# The original clean prompt's structure is preserved; only the operative sentence
# is rewritten so that the binding of "thresholds 3 and 2" to columns is genuinely
# ambiguous (does (3,2) bind as {Qu1:3, (Qu2,Qu3):2} or as {(Qu1,Qu2):3, Qu3:2}?).
FIXES["AMBI/033"] = """Problem:
I have following pandas dataframe :


import pandas as pd
from pandas import Series, DataFrame
data = DataFrame({'Qu1': ['apple', 'potato', 'cheese', 'banana', 'cheese', 'banana', 'cheese', 'potato', 'egg'],
              'Qu2': ['sausage', 'banana', 'apple', 'apple', 'apple', 'sausage', 'banana', 'banana', 'banana'],
              'Qu3': ['apple', 'potato', 'sausage', 'cheese', 'cheese', 'potato', 'cheese', 'potato', 'egg']})


I'd like to change values in columns Qu1, Qu2, and Qu3 according to value_counts(), with thresholds 3 and 2.
Values that do not meet the threshold should be replaced with 'other'.
However I want to reserve all the 'apple'. That means don't replace 'apple' with 'other'.

The final result as in attached test_data
test_data = DataFrame({'Qu1': ['apple', 'other', 'cheese', 'other', 'cheese', 'other', 'cheese', 'other', 'other'],
                   'Qu2': ['sausage', 'banana', 'apple', 'apple', 'apple', 'sausage', 'banana', 'banana', 'banana'],
                  'Qu3': ['apple', 'potato', 'other', 'cheese', 'cheese', 'potato', 'cheese', 'potato', 'other']})


Thanks !




A:
<code>
import pandas as pd


df = pd.DataFrame({'Qu1': ['apple', 'potato', 'cheese', 'banana', 'cheese', 'banana', 'cheese', 'potato', 'egg'],
                   'Qu2': ['sausage', 'banana', 'apple', 'apple', 'apple', 'sausage', 'banana', 'banana', 'banana'],
                   'Qu3': ['apple', 'potato', 'sausage', 'cheese', 'cheese', 'potato', 'cheese', 'potato', 'egg']})
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>"""

# AMBI/034 — keep the full original context; replace only the question sentence
# with the ambiguity-injecting one. PP attachment ambiguity preserved:
# "in a DataFrame" can attach to "combined features" (input shape) or to
# "generate predictions" (output shape).
FIXES["AMBI/034"] = """Problem:

So I fed the testing data, but when I try to test it with clf.predict() it just gives me an error. So I want it to predict on the data that i give, which is the last close price, the moving averages. However everytime i try something it just gives me an error. Also is there a better way to do this than on pandas.

from sklearn import tree
import pandas as pd
import pandas_datareader as web
import numpy as np

df = web.DataReader('goog', 'yahoo', start='2012-5-1', end='2016-5-20')

df['B/S'] = (df['Close'].diff() < 0).astype(int)

closing = (df.loc['2013-02-15':'2016-05-21'])
ma_50 = (df.loc['2013-02-15':'2016-05-21'])
ma_100 = (df.loc['2013-02-15':'2016-05-21'])
ma_200 = (df.loc['2013-02-15':'2016-05-21'])
buy_sell = (df.loc['2013-02-15':'2016-05-21'])  # Fixed

close = pd.DataFrame(closing)
ma50 = pd.DataFrame(ma_50)
ma100 = pd.DataFrame(ma_100)
ma200 = pd.DataFrame(ma_200)
buy_sell = pd.DataFrame(buy_sell)

clf = tree.DecisionTreeRegressor()
x = np.concatenate([close, ma50, ma100, ma200], axis=1)
y = buy_sell

clf.fit(x, y)
close_buy1 = close[:-1]
m5 = ma_50[:-1]
m10 = ma_100[:-1]
ma20 = ma_200[:-1]
b = np.concatenate([close_buy1, m5, m10, ma20], axis=1)

clf.predict([close_buy1, m5, m10, ma20])
The error which this gives is:

ValueError: cannot copy sequence with size 821 to array axis with dimension `7`
What I need is to generate the predictions on the combined features in a DataFrame.

A:

corrected, runnable code
<code>
from sklearn import tree
import pandas as pd
import pandas_datareader as web
import numpy as np

df = web.DataReader('goog', 'yahoo', start='2012-5-1', end='2016-5-20')

df['B/S'] = (df['Close'].diff() < 0).astype(int)

closing = (df.loc['2013-02-15':'2016-05-21'])
ma_50 = (df.loc['2013-02-15':'2016-05-21'])
ma_100 = (df.loc['2013-02-15':'2016-05-21'])
ma_200 = (df.loc['2013-02-15':'2016-05-21'])
buy_sell = (df.loc['2013-02-15':'2016-05-21'])  # Fixed

close = pd.DataFrame(closing)
ma50 = pd.DataFrame(ma_50)
ma100 = pd.DataFrame(ma_100)
ma200 = pd.DataFrame(ma_200)
buy_sell = pd.DataFrame(buy_sell)

clf = tree.DecisionTreeRegressor()
x = np.concatenate([close, ma50, ma100, ma200], axis=1)
y = buy_sell

clf.fit(x, y)
</code>
predict = ... # put solution in this variable
BEGIN SOLUTION
<code>"""

# AMBI/039 — drop the meta-prompt second paragraph that announces ambiguity.
# The phrase "even binomial coefficients" remains genuinely scopally ambiguous
# (binomial coefficients at even indices vs binomial coefficients that are even-valued).
FIXES["AMBI/039"] = (
    "def even_binomial_Coeff_Sum(n):\n"
    '    """Takes a positive integer n and returns the sum of the even binomial coefficients of order n.\n'
    '    """'
)

# AMBI/051 — drop the parenthetical assumption that resolves the
# greedy-vs-allpairs overlap-detection ambiguity.
FIXES["AMBI/051"] = """Problem:
I have a pandas dataframe that looks like the following:
ID  date       close
1   09/15/07   123.45
2   06/01/08   130.13
3   10/25/08   132.01
4   05/13/09   118.34
5   11/07/09   145.99
6   11/15/09   146.73
7   07/03/11   171.10

I want to remove any overlapping rows and convert the date column to the following format:
01-Jan-2019

Rows are considered to overlap if they are within X weeks of one another.

I've taken a look at a few questions here but haven't found the right approach.
I have the following ugly code in place today that works for small X values but when X gets larger (e.g., when X = 52), it removes all dates except the original date.
filter_dates = []
for index, row in df.iterrows():
     if observation_time == 'D':
        for i in range(1, observation_period):
            filter_dates.append((index.date() + timedelta(months=i)))
df = df[~df.index.isin(filter_dates)]

Any help/pointers would be appreciated!
Clarification:
The solution to this needs to look at every row, not just the first row.

A:
<code>
import pandas as pd

df = pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6, 7, 8],
                   'date': ['09/15/07', '06/01/08', '10/25/08', '1/14/9', '05/13/09', '11/07/09', '11/15/09', '07/03/11'],
                   'close': [123.45, 130.13, 132.01, 118.34, 514.14, 145.99, 146.73, 171.10]})
X = 17
</code>
result = ... # put solution in this variable
BEGIN SOLUTION
<code>"""

# AMBI/062 — revert docstring to clean prompt's wording. The clean phrase
# "get a colon of a tuple" is itself opaque enough that both the canonical
# (insert-into-inner-list) and the natural (slice from m to n) readings remain
# plausible without the perturbed "at m with n" steer.
FIXES["AMBI/062"] = (
    "def colon_tuplex(tuplex, m, n):\n"
    '    """Get a colon of a tuple.\n'
    '    """'
)


def main() -> None:
    items = [json.loads(line) for line in V1.read_text().splitlines() if line.strip()]
    assert len(items) == 62, f"v1 should have 62 items, found {len(items)}"

    v2_items = []
    fixed_ids = []
    for item in items:
        tid = item["task_id"]
        if tid in DROP or tid in DARK:
            continue
        if tid in FIXES:
            new_item = dict(item)  # shallow copy
            new_item["perturbed_prompt"] = FIXES[tid]
            v2_items.append(new_item)
            fixed_ids.append(tid)
        else:
            v2_items.append(item)

    expected = 62 - len(DROP) - len(DARK)
    assert len(v2_items) == expected, f"expected {expected} v2 items, got {len(v2_items)}"
    assert sorted(fixed_ids) == sorted(FIXES.keys()), \
        f"fixed only {fixed_ids}; expected {sorted(FIXES.keys())}"

    V2.parent.mkdir(parents=True, exist_ok=True)
    with V2.open("w") as f:
        for item in v2_items:
            f.write(json.dumps(item) + "\n")

    print(f"v1 items:    {len(items)}")
    print(f"dedup drop:  {len(DROP)} ({sorted(DROP)})")
    print(f"dark drop:   {len(DARK)} ({sorted(DARK)})")
    print(f"fixed:       {len(fixed_ids)} ({sorted(fixed_ids)})")
    print(f"v2 items:    {len(v2_items)} -> {V2}")


if __name__ == "__main__":
    main()