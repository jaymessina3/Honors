import requests
import lxml.html as lh
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
#a = requests.get("https://www.basketball-reference.com/players/j/jamesle01/gamelog-advanced/2014/")
a = requests.get("https://www.basketball-reference.com/players/j/butleji01/gamelog/2014/")
soup = BeautifulSoup(a.text, 'lxml')

url = 'https://www.basketball-reference.com/players/j/jamesle01/gamelog/2014'
url = url.replace("01", "02", 1)
print(url)
#Create a handle, page, to handle the contents of the website
page = requests.get(url)
#Store the contents of the website under doc
doc = lh.fromstring(page.content)
#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//table//thead//tr')
rows = soup.findAll('table', {'class': 'row_summable sortable stats_table now_sortable'})

tr = soup.find('div', {"class":"table_wrapper"}, "tbody")
tb = soup.find("tbody")
#print(tb)
#print(tb.textcontent())
row = tb.findAll("tr")
#r = row.findAll("td", {"data-stat":"fga"})

#sales = [('Jones LLC', 150, 200, 50), ('Alpha Co', 200, 210, 90), ('Blue Inc', 140, 215, 95)]
labels = ['GM', 'Date', 'FGA', 'FTA']
#df = pd.DataFrame.from_records(sales, columns=labels)

bigL = []
#print(row)
for x in row:
    items = x.findAll("td")
    #items = x.findAll("td")

    counter = 0
    L = []
    # filter by id with element, filter td that have the certain id
    for y in items:
        #print(y)
        #rex = re.compile(r'<td.*?>(.*?)</td>',re.S|re.M)
        #match = rex.match(str(y))
        #cleanr = re.compile('<.*?>')
        #cleantext = re.sub(cleanr, '', y)
        #print(y.text)
        if (counter == 0 or counter == 1 or counter == 10 or counter == 16):
            
            L.append(y.text)
        counter+=1
    
    bigL.append(L)
    #print(bigL)
df = pd.DataFrame.from_records(bigL, columns=labels)
df.dropna(thresh=3)
writer = pd.ExcelWriter('output.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()
#book = openpyxl.Workbook()
#book = openpyxl.load_workbook('output.xlsx')
#sheet = book.get_sheet_by_name('Sheet1')  
#for loop through and delete none rows 
print(df)
        
    
    #print(x)

    
    

#tr_elements = doc.xpath("//div[@id="div_pgl_basic"]")


'''
td_list = tr.find_all("td")
print(td_list)
#print([len(T) for T in tr_elements[:12]])
print([len(T) for T in tr_elements[:12]])

#Create empty list
col=[]
i=0
#For each row, store each first element (header) and an empty list
for t in tr_elements[0]:
    i+=1
    name=t.text_content()
    print ('%d:"%s"'%(i,name))
    col.append((name,[]))
    
#Since out first row is the header, data is stored on the second row onwards
for j in range(1,len(tr_elements)):
    #T is our j'th row
    T=tr_elements[j]
    
    #If row is not of size 10, the //tr data is not from our table 
    if len(T)!=9:
        break
    
    #i is the index of our column
    i=0
    
    #Iterate through each element of the row
    for t in T.iterchildren():
        data=t.text_content() 
        #Check if row is empty
        if i>0:
        #Convert any numerical value to integers
            try:
                data=int(data)
            except:
                pass
        #Append the data to the empty list of the i'th column
        col[i][1].append(data)
        #Increment i for the next column
        i+=1
print([len(C) for (title,C) in col])

'''