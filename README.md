# gulls-postprocessing
A repository for examples of gulls postprocessing, visualizations, and utility functions

# Using the repository
1. Make a fork of the repository to your own github account
2. Clone the fork to your machine
3. Load the notebooks (note that the notebooks assume that you have run gulls_reduce.py and are starting with det and out HDF5 files)

# Contributing
You can contribute a notebook or utility function via a pull request (fork the repos. If you do, please follow the following guidelines:

- In the first cell of the notebook please put a brief summary of the purpose of the noteook, who wrote it, some contact details, and any other relevant information (e.g., how to cite etc.).
- gulls output is large, so please don't commit the data to the repository. Instead, if you are willing, please, put the data where someone can find it and add a link to it in the first cell.
- Uncommented/documented code is better than no code, but where possible please make use of markup and comments to explain what is being done. This is especially important for utility functions.
- If your notebook is a tutorial, please name it as tutorial_...
