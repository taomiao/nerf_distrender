class Scene:
    def __init__(self, aabb, model_name):
        self.aabb = aabb
        self.model_name = model_name
        self.child_scenes = []


class BlockScene(Scene):
    def __init__(self, aabb, model_name, row, col):
        super().__init__(aabb, model_name)
        self.total_block_num = 0
        self.init_by_row_col(row, col)

    def init_by_row_col(self, row, col):
        self.total_block_num = row*col
        left_down = self.aabb[0]
        right_up = self.aabb[1]
        block_w = right_up[0] - left_down[0]
        block_h = right_up[1] - left_down[1]
        for i in range(row):
            for j in range(col):
                child_aabb = left_down + i * [block_w, 0, 0] + j * [0, block_h, 0]
                child_model_name = self.model_name+f"_{i}x{j}"
                child_scene = Scene(child_aabb, child_model_name)
                self.child_scenes.append(child_scene)

