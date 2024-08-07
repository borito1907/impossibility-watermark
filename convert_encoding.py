import codecs

blockSize = 1048576
with codecs.open("data/wqe_watermark_samples.csv","r",encoding="cp1252") as sourceFile:
    with codecs.open("data/wqe_watermark_samples_converted.csv","w",encoding="UTF-8") as targetFile:
        while True:
            contents = sourceFile.read(blockSize)
            if not contents:
                break
            targetFile.write(contents)