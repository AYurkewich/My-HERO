from firebase import firebase

firebase = firebase.FirebaseApplication('https://hero-d6297.firebaseio.com/')
# data = { 'Name': 'Vivek',
#           'RollNo': 1,
#           'Percentage': 76.02
#           }
#result = firebase.get('/MyoCommand', None)
#use put not post
firebase.put('', 'MyoCommand', 'open')
#firebase.put('hero-d6297' hello', 'hi', '2')
#print(result)

