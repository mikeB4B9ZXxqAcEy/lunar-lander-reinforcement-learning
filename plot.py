import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-poster')
# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

XLABEL = r'\textbf{Attempt Number}'
YLABEL = r'\textbf{Score}'
REQUIRED_SCORE = 200
##############################################################################
def plot_training_scores():
    df = pd.read_csv('./results/train_scores.csv')
    df['attempt_num'] = range(1, len(df) + 1)


    print(df.head())


    fig = plt.figure()
    ax = plt.axes()

    ax.plot(df['attempt_num'], df['score'], color='tab:blue')

    # Draw a red line across the required score
    ax.axhline(y=REQUIRED_SCORE, color='tab:red', linestyle='-')

    # Draw a green line where the agent first gets the required score
    # Returns the attempt number (+ 1 because we're one based)
    first_successful_landing = df[df['score'].gt(199.99)].index[0] + 1
    print(f"First Successful Landing is attempt # {first_successful_landing}")
    ax.axvline(x=first_successful_landing, color='tab:green', linestyle='--')

    plt.xlim((1, df['attempt_num'].max()))
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.title(r'\textbf{Learning Results Over Attempts}')
    plt.savefig('results/train_scores.pdf')
    plt.show()


def plot_MA_train_scores():
    df = pd.read_csv('./results/train_scores.csv')
    df['attempt_num'] = range(1, len(df) + 1)

    print(df.head())

    fig = plt.figure()
    ax = plt.axes()

    score_mean = pd.Series(df['score']).rolling(window=5).mean()

    ax.plot(df['attempt_num'], score_mean, color='tab:blue')

    # Draw a red line across the required score
    ax.axhline(y=REQUIRED_SCORE, color='tab:red', linestyle='-')

    # Draw a green line where the agent first gets the required score
    # Returns the attempt number (+ 1 because we're one based)
    first_successful_landing = score_mean[score_mean.gt(199.99)].index[0] + 1
    print(f"First Successful Landing is attempt # {first_successful_landing}")
    ax.axvline(x=first_successful_landing, color='tab:green', linestyle='--')

    plt.xlim((1, df['attempt_num'].max()))
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.title(r'\textbf{Learning Results Over Attempts}')
    plt.savefig('results/train_scores_MA.pdf')
    plt.show()

#############################################################################

def plot_play_scores():
    df = pd.read_csv('./results/play_scores.csv')
    df['attempt_num'] = range(1, len(df) + 1)

    print(df.head())

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(df['attempt_num'], df['score'], color='tab:blue')

    # Draw a red line across the required score
    ax.axhline(y=REQUIRED_SCORE, color='tab:red', linestyle='-')

    plt.xlim((1, df['attempt_num'].max()))
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.title(r'\textbf{Playing Results Over Attempts}')
    plt.savefig('results/play_scores.pdf')
    plt.show()

def plot_MA_play_scores():
    df = pd.read_csv('./results/play_scores.csv')
    df['attempt_num'] = range(1, len(df) + 1)

    print(df.head())

    fig = plt.figure()
    ax = plt.axes()

    score_mean = pd.Series(df['score']).rolling(window=20).mean()

    ax.plot(df['attempt_num'], score_mean, color='tab:blue')

    # Draw a red line across the required score
    ax.axhline(y=REQUIRED_SCORE, color='tab:red', linestyle='-')

    plt.xlim((1, df['attempt_num'].max()))
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.title(r'\textbf{Playing Results Over Attempts}')
    plt.savefig('results/play_scores_MA.pdf')
    plt.show()

plot_training_scores()
plot_play_scores()
plot_MA_train_scores()
plot_MA_play_scores()
