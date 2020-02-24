# Applying memoization in python
def deletion_distance(str1, str2):
    dt = {}
    return deletion_distance_aux(str1, str2, dt)

def deletion_distance_aux(str1, str2, dt):
    # base case - return the length of the other if one is zero
    
    if (str1, str2) in dt:
        return dt[str1, str2]
    if len(str1) == 0:
        return len(str2)
    if len(str2) == 0:
        return len(str1)
    
    if str1[0] == str2[0]:
        result = 0 + deletion_distance_aux(str1[1:], str2[1:], dt)
    else:
        result = 1 + min(deletion_distance_aux(str1[1:], str2, dt),
                         deletion_distance_aux(str1, str2[1:], dt))   
    
    dt[str1, str2] = result
    
    return result



#def deletion_distance(str1, str2):
#    if (str1, str2) in aux_dict:
#        return aux_dict[str1, str2]
#    if not str1:
#        return len(str2)
#    elif not str2:
#        return len(str1)
#    elif str1[-1] == str2[-1]:
#        result = deletion_distance(str1[:-1],str2[:-1])
#    else:
#        result = 1 + min(
#            deletion_distance(str1, str2[:-1]),
#            deletion_distance(str1[:-1], str2),
#        )
#    aux_dict[str1, str2] = result
#    return result
#
#aux_dict = {}