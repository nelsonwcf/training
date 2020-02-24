# note: a-97, z-122

def decrypt(word):
    
    carry_over = 1
    prev_value = 1
    decr = ''
    
    for c in word:
        n = ord(c)
        prev_value = n
        n -= carry_over
        while (n < 97):
            n += 26
        decr = ''.join((decr, chr(n)))
        carry_over = prev_value
    
    return decr