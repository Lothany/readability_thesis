class Fragment:
    def __init__(self, original, index):
        self.original = original
        self.index = index
        self.grade_scores = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

    def __repr__(self):
        return str(self.grade_scores)

    def set_score(self, grade, score):
        self.grade_scores[grade] = score

    def show(self):
        return self.original

    def to_dict(self):
        return {
            1: self.grade_scores[1],
            2: self.grade_scores[2],
            3: self.grade_scores[3],
            4: self.grade_scores[4],
            5: self.grade_scores[5],
            6: self.grade_scores[6]
        }
