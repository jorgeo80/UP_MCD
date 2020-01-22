#################################################################
#################################################################
####   Intermediate Python for Data Science                 #####
#################################################################
#################################################################

###########################
####   Ejercicio 1    #####
###########################
# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']
# Print each list item in flash using a for loop
for person in flash:
    print(person)
# Cnreate an iterator for flash: superspeed
for superspeed in flash:
    superspeed=iter(superspeed)
# Print each item from the iterator
print(next(superspeed))
print(next(superspeed))
print(next(superspeed))
print(next(superspeed))

###########################
####   Ejercicio 2    #####
###########################
# Create an iterator for range(3): small_value
small_value = iter(range(3))
# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))
# Loop over range(3) and print the values
for num in range(3):
    print(num)
# Create an iterator for range(10 ** 100): googol
googol= iter(range(10 ** 100))
# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))

###########################
####   Ejercicio 3    #####
###########################
# Create a range object: values
values=range(10,21)
# Print the range object
print(values)
# Create a list of integers: values_list
values_list=list(values)
# Print values_list
print(values_list)
# Get the sum of values: values_sum
values_sum =sum(values)
# Print values_sum
print(values_sum)

###########################
####   Ejercicio 4    #####
###########################
# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']
# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))
# Print the list of tuples
print(mutant_list)
# Unpack and print the tuple pairs
for index1,value1 in enumerate(mutants):
    print(index1, value1)
# Change the start index
for index2,value2 in enumerate(mutants,start=1):
    print(index2, value2)

###########################
####   Ejercicio 5    #####
###########################
# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))
# Print the list of tuples
print(mutant_data)
# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)
# Print the zip object
print(mutant_zip)
# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)

###########################
####   Ejercicio 6    #####
###########################
# Create a zip object from mutants and powers: z1
z1 =zip(mutants,powers)
# Print the tuples in z1 by unpacking with *
print(*z1)
# Re-create a zip object from mutants and powers: z1
z1=zip(mutants,powers)
# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)
# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)

###########################
####   Ejercicio 7    #####
###########################
# Initialize an empty dictionary: counts_dict
counts_dict={}
# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv',chunksize=10):
    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1
# Print the populated dictionary
print(counts_dict)


###########################
####   Ejercicio 8    #####
###########################
# Define count_entries()
def count_entries(csv_file,c_size,colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}
# Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file,chunksize=c_size):
        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1
    # Return counts_dict
    return counts_dict
# Call count_entries(): result_counts
result_counts = count_entries('tweets.csv',10,'lang')
# Print result_counts
print(result_counts)

###########################
####   Ejercicio 9    #####
###########################


###########################
####   Ejercicio 10   #####
###########################

###########################
####   Ejercicio 11   #####
###########################

###########################
####   Ejercicio 12   #####
###########################

###########################
####   Ejercicio 13   #####
###########################

###########################
####   Ejercicio 14   #####
###########################

###########################
####   Ejercicio 15   #####
###########################

###########################
####   Ejercicio 16   #####
###########################

###########################
####   Ejercicio 17   #####
###########################

###########################
####   Ejercicio 18   #####
###########################

###########################
####   Ejercicio 19   #####
###########################

###########################
####   Ejercicio 20   #####
###########################

###########################
####   Ejercicio 21   #####
###########################

###########################
####   Ejercicio 22   #####
###########################

###########################
####   Ejercicio 23   #####
###########################

###########################
####   Ejercicio 24   #####
###########################

###########################
####   Ejercicio 25   #####
###########################

###########################
####   Ejercicio 26   #####
###########################

###########################
####   Ejercicio 27   #####
###########################

###########################
####   Ejercicio 28   #####
###########################

###########################
####   Ejercicio 29   #####
###########################

###########################
####   Ejercicio 30   #####
###########################

