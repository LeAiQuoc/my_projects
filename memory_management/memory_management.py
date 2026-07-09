

class Arena:
    """
    Represents a large chunk of memory (max size 256KB), managing multiple pools.
    """
    MAX_SIZE = 256 * 1024  # 256KB

    def __init__(self):
        self.pools = []  # List of Pool objects
        self.current_size = 0  # Track used memory in bytes

    def add_pool(self, pool):
        """
        Adds a pool to the arena if there is enough space.
        """
        if self.current_size + pool.size <= self.MAX_SIZE:
            self.pools.append(pool)
            self.current_size += pool.size
            return True
        return False

class Pool:
    """
    Represents a chunk of memory (max size 4KB) containing blocks.
    """
    MAX_SIZE = 4 * 1024  # 4KB

    def __init__(self, name):
        self.name = name
        self.blocks = []  # List of Block objects
        self.size = 0

    def add_block(self, block):
        """
        Adds a block to the pool if there is enough space.
        """
        if self.size + block.size <= self.MAX_SIZE:
            self.blocks.append(block)
            self.size += block.size
            return True
        return False

    def remove_block(self, block):
        """
        Removes a block from the pool
        """
        if block in self.blocks:
            self.blocks.remove(block)
            self.size -= block.size
            return True
        return False

class Block:
    """
    Represents a small unit of memory to store a value.
    """
    def __init__(self, value):
        self.value = value
        self.size = self._calculate_size()

    def set_value(self, value):
        self.value = value
        self.size = self._calculate_size()

    def _calculate_size(self):
        # Calculate size based on data type
        if isinstance(self.value, int):
            return 32
        elif isinstance(self.value, str):
            return 64 + len(self.value)
        elif hasattr(self.value, '__dict__'):
            return 128
        else:
            return 64

class MemoryManager:
    """
    Manages the interaction between arenas, pools, and blocks.
    Handles allocation and deallocation of blocks.
    """
    def __init__(self):
        self.arena = Arena()
        self.pool_counter = 0
        self.free_blocks = []

    def allocate_block(self, value):
        """
        Allocates a block with the given value.
        Reuses a free block if available, otherwise creates a new block.
        """
        # Check if we can reuse a free block
        if self.free_blocks:
            block = self.free_blocks.pop()
            block.set_value(value)
        else:
            block = Block(value)
        
        # Try to add to existing pool
        for pool in self.arena.pools:
            if pool.add_block(block):
                return block
        
        # Need to create new pool
        new_pool = Pool(f"Pool_{self.pool_counter}")
        self.pool_counter += 1
        
        if new_pool.add_block(block):
            if self.arena.add_pool(new_pool):
                return block
            else:
                raise MemoryError("Arena is full. Cannot allocate new pool.")
        
        raise MemoryError("Block too large to fit in a new pool.")

    def deallocate_block(self, block):
        """
        Deallocates a block from its pool and adds it to the free block list for reuse.
        """
        for pool in self.arena.pools:
            if block in pool.blocks:
                pool.remove_block(block)
                self.free_blocks.append(block)
                return True
        return False

    def get_stats(self):
        """
        Returns statistics about memory usage
        """
        num_pools = len(self.arena.pools)
        num_blocks = sum(len(pool.blocks) for pool in self.arena.pools)
        num_free_blocks = len(self.free_blocks)
        arena_used = self.arena.current_size
        
        return {
            'pools_in_use': num_pools,
            'blocks_in_use': num_blocks,
            'free_blocks': num_free_blocks,
            'arena_used_bytes': arena_used
        }


# Example usage and testing
if __name__ == "__main__":
    mm = MemoryManager()
    
    print("\n--- Running basic tests ---")
    # Testing
    # Fill a pool with int blocks
    blocks = []
    for i in range(130):
        blocks.append(mm.allocate_block(i))
    print(f"After filling pool: {mm.get_stats()}")

    # Deallocate some blocks
    for b in blocks[:10]:
        mm.deallocate_block(b)
    print(f"After deallocation: {mm.get_stats()}")

    # Allovate more blocks (reuse free blocks)
    for i in range(10):
        mm.allocate_block(i * 100)
    print(f"After reusing free blocks: {mm.get_stats()}")

    # Store an integer
    int_block = mm.allocate_block(42)
    print(f"Allocated int block: {int_block.value}, size: {int_block.size}")

    # Store a string
    str_block = mm.allocate_block("Hello, memory management!")
    print(f"Allocated str block: {str_block.value}, size: {str_block.size}")

    # Store a custom object
    class CustomObject:
        def __init__(self, data):
            self.data = data

    obj = CustomObject([1, 2, 3])
    obj_block = mm.allocate_block(obj)
    print(f"Allocated object block: {obj_block.value.data}, size: {obj_block.size}")

    # Show stats
    stats = mm.get_stats()
    print(f"Memory stats: {stats}")