from abc import abstractmethod
class BatchScheduler:
    @abstractmethod
    def sample_to_scene(self, samples, scene):
        pass
class SimpleBatchScheduler(BatchScheduler):
    def sample_to_scene(self, samples, scene):
        batched_samples = []
        for sample in samples:
            if scene.contain(sample):
                batched_samples.append(sample)
        return batched_samples

class BlockSceneBatchScheduler(BatchScheduler):
    def sample_to_scene(self, samples, scene):
        pass