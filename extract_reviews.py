import os,sys,re

def crawlData(sourceDir, destDir):
    sourceFiles = os.listdir(sourceDir)

    html_tags = re.compile(r'<[^>]+>')
    for file in sourceFiles:
        filepath = sourceDir+'/'+file;

        if file.endswith(".txt"):
            inFile = open(filepath,'r')
            print(inFile)
            data = ""
            result = ""
            rating = 0
            goodReview = False
            for line in inFile:
                foundreview = line.find('<i class="star-img',0)

                if foundreview > -1:
                    found25review = line.find('title="2.5 star rating">')
                    found3review = line.find('title="3.0 star rating">')
                    found35review = line.find('title="3.5 star rating">')
                    found4review = line.find('title="4.0 star rating">')
                    found45review = line.find('title="4.5 star rating">')
                    found5review = line.find('title="5.0 star rating">')
                    if (found5review > -1) or (found4review > -1) or (found45review > -1) or (found35review > -1) or (found3review > -1) or (found25review > -1):
                        goodReview = True
                    else:
                        goodReview = False

                foundreview = line.find('<p itemprop="description" lang="en">',0)

                if foundreview > -1:
                    if goodReview:
                        result = result+' '+html_tags.sub(' ',line)
                        
                    goodReview = False

            outFile = open(destDir+'/'+file,'w')
            outFile.write(result)
            outFile.close()
            inFile.close()

if __name__ == "__main__":
    crawlData(str(sys.argv[1]),str(sys.argv[2]))