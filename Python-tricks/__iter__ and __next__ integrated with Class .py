# The code below is to test "class" integrated with __iter__ and __next__ 

class MyNumbers:
  def __iter__(self):
    self.a = 1
    
    
    '''
    for better unerstand "__iter__ " and "__next__" 
    run below (#1 and #2 and #3 line)
    '''
    
    print("what is initial:  ",self.a) #1

   
    return self
 
  def __next__(self):
    x = self.a
    self.a += 1
    '''
    for better unerstand "__iter__ " and "__next__"  
    run below code and __iter__  “print”  (#1)  if you want to see the exact order of the excute process
    '''
    #2
    print("self.a is: ",self.a)  # test the effect  of __next__ 
    print("currunt x is: ",x)       # test when the self take effect 
    #2 end 
    return x
   
 
myclass = MyNumbers()
myiter = iter(myclass)


print("this is to test what happened before __next__ taking effect ### ")  #3

print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))


# inspired by the www.runoob.com