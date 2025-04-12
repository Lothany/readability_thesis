import foreignator
foreignator.mass_identify("run/english.txt")

# foreignator.mass_identify("../txt/normalized/g2/5.txt")
# foreignator.mass_identify("../txt/normalized/g2/11.txt")
# foreignator.mass_identify("../txt/normalized/g2/16.txt")
# foreignator.mass_identify("../txt/normalized/g2/56.txt")
# foreignator.mass_identify("../txt/normalized/g2/67.txt")
# foreignator.mass_identify("../txt/normalized/g2/68.txt")
# foreignator.mass_identify("../txt/normalized/g2/69.txt")
# foreignator.mass_identify("../txt/normalized/g2/99.txt")
# foreignator.mass_identify("../txt/normalized/g2/102.txt")

from word_embedding import WordEmbedding, LexicalMetadata, TraditionalMetadata

wb = WordEmbedding("txt/normalized/g1/19.txt")
wb.toJSON()
