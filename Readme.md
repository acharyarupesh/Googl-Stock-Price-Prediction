INFO7374-Machine Learning in Finance Final Project

Note: Please keep the directory as it is. Try running the Notebook files for better view of results. If you don't have python installed you can still access our code and results using HTML files of the same. We have also provided the python files for the same.

Project Requirements:
1. Python and related libraries
2. Anaconda
3. Jupyter

To run the project:
1. Run Data_Preparation.ipynb. Make sure you keep all the excel files in the same files.

2. Pricing Models (Random_Forest.ipynb):
Note: You will need to install prettytable, command: pip install prettytable. 
Uncomment the things which are in .fit() functions.
	a) AR1 Model
	b) Fama French 5-Factor
	c) Moving Average Model
	d) All Factors Model
	Note: Again comment the previously commented code, since it will generate ouput in each iteration
	e) Random Forest with bootstrapping
3. Returns Models (3 files):
	a) CAPM Returns (CAPM.ipynb)
	b) Kalman Filter Returns (Kalman.ipynb)
	c) GARCH Returns (Garch.ipynb)

4. Trading Strategies (2 files) 
	1st File:(Trading_Strategy.ipynb)
		a. Day Trading
		b. Long Short Trading
		c. Buy Hold
	2nd File:(MovingAverage.ipynb)
		a. Exponential Moving Average Trading Strategy

	