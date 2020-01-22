#################################################################
#################################################################
####   Intermediate Python for Data Science                 #####
#################################################################
#################################################################

###########################
####   Ejercicio 1    #####
###########################
# Print the last item from year and pop
print(year[-1])
print(pop[-1])

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year,pop)


# Display the plot with plt.show()
plt.show()

###########################
####   Ejercicio 2    #####
###########################
# Print the last item of gdp_cap and life_exp
print(gdp_cap[-1])
print(life_exp[-1])


# Make a line plot, gdp_cap on the x-axis, life_exp on the y-axis
plt.plot(gdp_cap,life_exp)

# Display the plot
plt.show()

###########################
####   Ejercicio 3    #####
###########################
# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()

###########################
####   Ejercicio 4    #####
###########################
# Import package
import matplotlib.pyplot as plt

# Build Scatter plot
plt.scatter(pop,life_exp)

# Show plot
plt.show()

###########################
####   Ejercicio 5    #####
###########################
# Create histogram of life_exp data
plt.hist(life_exp)
# Display histogram
plt.show()

###########################
####   Ejercicio 6    #####
###########################
# Build histogram with 5 bins
plt.hist(life_exp,bins=5)
# Show and clean up plot
plt.show()
plt.clf()
# Build histogram with 20 bins
plt.hist(life_exp,bins=20)
# Show and clean up again
plt.show()
plt.clf()

###########################
####   Ejercicio 7    #####
###########################
# Histogram of life_exp, 15 bins
plt.hist(life_exp,bins=15)
# Show and clear plot
plt.show()
plt.clf()
# Histogram of life_exp1950, 15 bins
plt.hist(life_exp1950,bins=15)
# Show and clear plot again
plt.show()
plt.clf()

###########################
####   Ejercicio 8    #####
###########################
# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 
# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'
# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)
# Add title
plt.title(title)
# After customizing, display the plot
plt.show()


###########################
####   Ejercicio 9    #####
###########################
# Scatter plot
plt.scatter(gdp_cap, life_exp)
# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
# Definition of tick_val and tick_lab
tick_val = [1000,10000,100000]
tick_lab = ['1k','10k','100k']
# Adapt the ticks on the x-axis
plt.xticks(tick_val,tick_lab)
# After customizing, display the plot
plt.show()

###########################
####   Ejercicio 10   #####
###########################
# Import numpy as np
import numpy as np
# Store pop as a numpy array: np_pop
np_pop=np.array(pop)
# Double np_pop
np_pop=np_pop*2
# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop)
# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])
# Display the plot
plt.show()

###########################
####   Ejercicio 11   #####
###########################
# Specify c and alpha inside plt.scatter()
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2,c=col,alpha=0.8)
# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])
# Show the plot
plt.show()

###########################
####   Ejercicio 12   #####
###########################
# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)
# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])
# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')
# Add grid() call
plt.grid(True)
# Show the plot
plt.show()

###########################
####   Ejercicio 13   #####
###########################
# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']
# Get index of 'germany': ind_ger
ind_ger=countries.index('germany')
# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])

###########################
####   Ejercicio 14   #####
###########################
# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']
# From string in countries and capitals, create dictionary europe
europe={'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo'}
# Print europe
print(europe)

###########################
####   Ejercicio 15   #####
###########################
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }
# Print out the keys in europe
print(europe.keys())
# Print out value that belongs to key 'norway'
print(europe['norway'])

###########################
####   Ejercicio 16   #####
###########################
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }
# Add italy to europe
europe.update({'italy':'rome'})
# Print out italy in europe
print('italy' in europe)
# Add poland to europe
europe.update({'poland':'warsaw'})
# Print europe
print(europe)

###########################
####   Ejercicio 17   #####
###########################
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw',
          'australia':'vienna' }
# Update capital of germany
europe.update({'germany':'berlin'})
# Remove australia
del(europe['australia'])
# Print europe
print(europe)

###########################
####   Ejercicio 18   #####
###########################
# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }
# Print out the capital of France
print(europe['france']['capital'])
# Create sub-dictionary data
data={'capital':'rome','population':59.83}
# Add data to europe under key 'italy'
europe.update({'italy':data})
# Print europe
print(europe)


###########################
####   Ejercicio 19   #####
###########################
# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
# Import pandas as pd
import pandas as pd
# Create dictionary my_dict with three key:value pairs: my_dict
my_dict={'country':names,'drives_right':dr,'cars_per_cap':cpc}
# Build a DataFrame cars from my_dict: cars
cars=pd.DataFrame(my_dict)
# Print cars
print(cars)

###########################
####   Ejercicio 20   #####
###########################
import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(cars_dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)


###########################
####   Ejercicio 21   #####
###########################
# Import pandas as pd
import pandas as pd
# Import the cars.csv data: cars
cars=pd.read_csv('cars.csv')
# Print out cars
print(cars)

###########################
####   Ejercicio 22   #####
###########################
# Import pandas as pd
import pandas as pd
# Fix import by including index_col
cars = pd.read_csv('cars.csv',index_col=0)
# Print out cars
print(cars)

###########################
####   Ejercicio 23   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out country column as Pandas Series
print(cars['country'])
# Print out country column as Pandas DataFrame
print(cars[['country']])
# Print out DataFrame with country and drives_right columns
print(cars[['country','drives_right']])

###########################
####   Ejercicio 24   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out first 3 observations
print(cars[0:3])
# Print out fourth, fifth and sixth observation
print(cars[3:6])

###########################
####   Ejercicio 25   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out observation for Japan
print(cars.loc['JAP'])
# Print out observations for Australia and Egypt
print(cars.loc[['AUS','EG']])

###########################
####   Ejercicio 26   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out drives_right value of Morocco
print(cars.loc[['MOR'],['drives_right']])
# Print sub-DataFrame
print(cars.loc[['RU','MOR'],['country','drives_right']])

###########################
####   Ejercicio 27   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out drives_right column as Series
print(cars.loc[:,'drives_right'])
# Print out drives_right column as DataFrame
print(cars.loc[:,['drives_right']])
# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:,['cars_per_cap','drives_right']])


###########################
####   Ejercicio 28   #####
###########################

# Comparison of booleans
print(True == False)
# Comparison of integers
print((-5*15) != 75)
# Comparison of strings
print('pyscript' == 'PyScript')
# Compare a boolean with an integer
print(True == 1)

###########################
####   Ejercicio 29   #####
###########################
# Comparison of integers
x = -3 * 6
print(x>=-10)
# Comparison of strings
y = "test"
print("test"<=y)
# Comparison of booleans
print(True>False)

###########################
####   Ejercicio 30   #####
###########################
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])
# my_house greater than or equal to 18
print(my_house>=18)
# my_house less than your_house
print(my_house<your_house)

###########################
####   Ejercicio 31   #####
###########################
# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(my_kitchen>10.0 and my_kitchen<18.0)

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen<14.0 or my_kitchen>17.0)

# Double my_kitchen smaller than triple your_kitchen?
print((my_kitchen*2)<(your_kitchen*3))

###########################
####   Ejercicio 32   #####
###########################
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])
# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house>18.5,my_house<10))
# Both my_house and your_house smaller than 11
print(np.logical_and(my_house<11,your_house<11))

###########################
####   Ejercicio 33   #####
###########################
# Define variables
room = "kit"
area = 14.0
# if statement for room
if room == "kit" :
    print("looking around in the kitchen.")
# if statement for area
if area>15:
    print('big place!')

###########################
####   Ejercicio 34   #####
###########################
# Define variables
room = "kit"
area = 14.0
# if-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
else :
    print("looking around elsewhere.")
# if-else construct for area
if area > 15 :
    print("big place!")
else :
    print('pretty small.')

###########################
####   Ejercicio 35   #####
###########################
# Define variables
room = "bed"
area = 14.0
# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")
# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area > 10 :
    print('medium size, nice!')
else :
    print("pretty small.")    

###########################
####   Ejercicio 36   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Extract drives_right column as Series: dr
dr=cars.loc[:,'drives_right']
# Use dr to subset cars: sel
sel=cars[dr]
# Print sel
print(sel)

###########################
####   Ejercicio 37   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Convert code to a one-liner
sel = cars[cars['drives_right']]
# Print sel
print(sel)

###########################
####   Ejercicio 38   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Create car_maniac: observations that have a cars_per_cap over 500
cpc=cars['cars_per_cap']
many_cars=cpc>500
car_maniac=cars[many_cars]
# Print car_maniac
print(car_maniac)

###########################
####   Ejercicio 39   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Create car_maniac: observations that have a cars_per_cap over 500
cpc=cars['cars_per_cap']
many_cars=cpc>500
car_maniac=cars[many_cars]
# Print car_maniac
print(car_maniac)

###########################
####   Ejercicio 40   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Import numpy, you'll need this
import numpy as np
# Create medium: observations with cars_per_cap between 100 and 500
medium=cars[np.logical_and(cars['cars_per_cap']>100,cars['cars_per_cap']<500)]
# Print medium
print(medium)

###########################
####   Ejercicio 41   #####
###########################
# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0:
        offset=offset-1
    else :
        offset=offset+1
    print(offset)

###########################
####   Ejercicio 42   #####
###########################
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
# Code the for loop
for x in areas:
    print(x)

###########################
####   Ejercicio 43   #####
###########################
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
# Change for loop to use enumerate()
for i, a in enumerate(areas) :
    print('room '+str(i)+': '+str(a))

###########################
####   Ejercicio 44   #####
###########################
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for index, area in enumerate(areas) :
    print("room " + str(index+1) + ": " + str(area))

###########################
####   Ejercicio 45   #####
###########################
# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
# Build a for loop from scratch
for x,y in house:
    print('the '+str(x)+' is '+str(y)+' sqm')

###########################
####   Ejercicio 46   #####
###########################
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for x, y in europe.items():
    print('the capital of '+str(x)+' is '+str(y))

###########################
####   Ejercicio 47   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Iterate over rows of cars
for x, y in cars.iterrows():
    print(x)
    print(y)

###########################
####   Ejercicio 48   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab+': '+str(row['cars_per_cap']))

###########################
####   Ejercicio 49   #####
###########################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)
# Use .apply(str.upper)
cars["COUNTRY"] = cars["country"].apply(str.upper)

###########################
####   Ejercicio 50   #####
###########################
# Import numpy as np
import numpy as np
# Set the seed
np.random.seed(123)
# Generate and print random float
print(np.random.rand())

###########################
####   Ejercicio 51   #####
###########################
# Import numpy and set seed
import numpy as np
np.random.seed(123)
# Use randint() to simulate a dice
print(np.random.randint(1,7))
# Use randint() again
print(np.random.randint(1,7))

###########################
####   Ejercicio 52   #####
###########################
# Import numpy and set seed
import numpy as np
np.random.seed(123)
# Starting step
step = 50
# Roll the dice
dice=np.random.randint(1,7)
# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice<=5 :
    step = step + 1
else :
    step = step + np.random.randint(1,7)
# Print out dice and step
print(dice)
print(step)

###########################
####   Ejercicio 53   #####
###########################
# Import numpy and set seed
import numpy as np
np.random.seed(123)
# Initialize random_walk
random_walk=[0]
# Complete the ___
for x in range(100) :
    # Set step: last element in random_walk
    step=random_walk[-1]
    # Roll the dice
    dice = np.random.randint(1,7)
    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)
    # append next_step to random_walk
    random_walk.append(step)
# Print random_walk
print(random_walk)
###########################
####   Ejercicio 54   #####
###########################
# Import numpy and set seed
import numpy as np
np.random.seed(123)
# Initialize random_walk
random_walk = [0]
for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)
    if dice <= 2:
        # Replace below: use max to make sure step can't go below 0
        step = max(step,1) - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)
    random_walk.append(step)
print(random_walk)

###########################
####   Ejercicio 55   #####
###########################
# Initialization
import numpy as np
np.random.seed(123)
random_walk = [0]
for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)
    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)
    random_walk.append(step)
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# Plot random_walk
plt.plot(random_walk)
# Show the plot
plt.show()

###########################
####   Ejercicio 56   #####
###########################
# Initialization
import numpy as np
np.random.seed(123)
# Initialize all_walks
all_walks=[]
# Simulate random walk 10 times
for i in range(10) :
    # Code from before
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    # Append random_walk to all_walks
    all_walks.append(random_walk)
# Print all_walks
print(all_walks)


###########################
####   Ejercicio 57   #####
###########################
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to Numpy array: np_aw
np_aw=np.array(all_walks)
# Plot np_aw and show
plt.plot(np_aw)
plt.show()
# Clear the figure
plt.clf()
# Transpose np_aw: np_aw_t
np_aw_t=np.transpose(np_aw)
# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()


###########################
####   Ejercicio 58   #####
###########################
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
all_walks = []
# Simulate random walk 250 times
for i in range(250) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        # Implement clumsiness
        if np.random.rand()<=0.001 :
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()

###########################
####   Ejercicio 59   #####
###########################
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
all_walks = []
# Simulate random walk 500 times
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
# Select last row from np_aw_t: ends
ends=np.array(np_aw_t[-1])
# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()












