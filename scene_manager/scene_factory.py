from scene_manager.scene import BlockScene


class SceneFactory:
    def create_4x6_block_scene(self):
        return BlockScene([[-18., -6., 0.], [-6., 6., 1.]], "shcity_1k", 4, 6)
