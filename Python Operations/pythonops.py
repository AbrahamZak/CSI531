'''
Python Operations

@author: Abraham Zakharov
'''

#part 1 
def test_print():
    print("This is a test statement.")
#part 2
def list_set_length():
    items_list =  [1, 2, 3, 4, 3, 2, 1] 
    items_set = set([1, 2, 3, 4, 3, 2, 1])
    print("Size of the list: " + str(len(items_list)))
    print("Size of the set: " + str(len(items_set)))
#part 3
def set_intersect(S,T):
    items_intersect = {x for x in S if x in T}
    return items_intersect
#part 4
def three_tuples(U):
    items_tuples = [(i,j,k) for i in U for j in U for k in U if i+j+k == 0]
    return items_tuples
#part 5    
def dict_init():
    mydict = {'Neo':'Keanu', 'Morpheus':'Laurence', 'Trinity':'Carrie-Anne'}
    return mydict

def dict_find(dlist, k):
    not_found = "NOT PRESENT"
    dict_list = [x.get(k, not_found) for x in dlist]
    return dict_list

#part 6
def file_line_count(filename):
    return len(open(filename).readlines())
    
#part 7
def make_inverse_index(strlist):
    dict_result_inverse_index = dict()
    for i, line in enumerate(open(strlist).readlines()):
        for word in line.split():
            if word.lower() in dict_result_inverse_index:
                dict_result_inverse_index[word.lower()].update({i})
            else:
                dict_result_inverse_index[word.lower()] = {i}
    return dict_result_inverse_index

def or_search(inverseIndex, query):
    query_breakdown = query.split()
    result_set = set.union(*[inverseIndex[word] for word in query_breakdown])
    return result_set
    
def and_search(inverseIndex, query):
    query_breakdown = query.split()
    result_set = set.intersection(*[inverseIndex[word] for word in query_breakdown])
    return result_set

if __name__ == '__main__':
    #part 1 test
    test_print()
    #part 2 test
    list_set_length()
    #part 3 test
    S = {1,2,3,4}
    T = {3,4,5,6}
    print("Set intersection: " + str(set_intersect(S,T)))
    #part 4 test
    U = {-4,-2, 1, 2, 5, 0}
    print("Tuples: " + str(three_tuples(U)))
    #part 5 test
    dict_0 = dict_init()
    dict_1 = {'Brand':'Honda', 'Car':'Civic'}
    dict_2 = {'Brand':'Apple', 'Model':'Macbook Pro'}
    dict_3 = {'Brand':'Starbucks', 'Type':'Coffee'}
    dict_4 = {'Name':'Abraham', 'Job':'Engineer'}
    dlist = [dict_0, dict_1, dict_2, dict_3, dict_4]
    k = "Brand"
    print("Dictionary find for key (" + k + "): " + str(dict_find(dlist,k)))
    #part 6 test
    filename = 'stories.txt'
    print("Lines: " + str(file_line_count(filename)))
    #part 7 test
    inverse_index = make_inverse_index(filename)
    print(inverse_index)
    inverse_index_or_search = or_search(inverse_index, "united states")
    print(inverse_index_or_search)
    inverse_index_or_search = or_search(inverse_index, "wall street")
    print(inverse_index_or_search)
    inverse_index_and_search = and_search(inverse_index, "united states")
    print(inverse_index_and_search)
    inverse_index_and_search = and_search(inverse_index, "wall street")
    print(inverse_index_and_search)
    
    
