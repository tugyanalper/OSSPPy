from __future__ import print_function
import timeit
import array
#import pyximport; pyximport.install()
#from OSSP_Dynamic_Array_cy import main

cy = timeit.timeit('OSSP_Dynamic_Array_cy.main()', setup='import OSSP_Dynamic_Array_cy', number=1)
# py = timeit.timeit('OSSP_GA.main()', setup='import OSSP_GA', number=1)
print(cy)
# print('Cython is %f faster than Python.'%(py/cy))