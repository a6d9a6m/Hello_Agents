"""测试记忆子系统"""

from src.memory import MemoryManager, MemoryType
from src.tools.builtin import MemoryTool, RAGTool


def test_memory_manager():
    """测试记忆管理器"""
    print("=== 测试记忆管理器 ===")
    
    manager = MemoryManager()
    
    # 测试存储记忆
    print("\n1. 存储工作记忆...")
    memory_id = manager.store(
        content="用户喜欢喝咖啡，每天早上一杯",
        memory_type=MemoryType.WORKING,
        user_id="user123",
        category="preference"
    )
    print(f"   存储成功，ID: {memory_id}")
    
    # 测试检索记忆
    print("\n2. 检索记忆...")
    memories = manager.retrieve("咖啡", limit=3)
    print(f"   找到 {len(memories)} 条相关记忆:")
    for i, memory in enumerate(memories, 1):
        print(f"   {i}. [{memory.memory_type.value}] {memory.content[:50]}...")
    
    # 测试获取上下文
    print("\n3. 获取上下文...")
    context = manager.retrieve_context("用户喜好", context_size=2)
    print(f"   上下文: {context[:100]}...")
    
    # 测试统计
    print("\n4. 获取统计信息...")
    stats = manager.get_stats()
    print(f"   统计: {stats}")
    
    print("\n[完成] 记忆管理器测试完成")


def test_memory_tool():
    """测试记忆工具"""
    print("\n=== 测试记忆工具 ===")
    
    tool = MemoryTool()
    
    # 测试存储
    print("\n1. 测试存储记忆...")
    result = tool.run(
        operation="store",
        content="项目截止日期是下周五",
        memory_type="working",
        metadata={"priority": "high", "project": "AI系统"}
    )
    print(f"   结果: {result.output}")
    print(f"   成功: {result.success}")
    
    # 测试检索
    print("\n2. 测试检索记忆...")
    result = tool.run(
        operation="retrieve",
        content="截止日期",
        memory_type="working",
        limit=3
    )
    print(f"   结果: {result.output[:100]}...")
    print(f"   成功: {result.success}")
    
    # 测试统计
    print("\n3. 测试获取统计...")
    result = tool.run(operation="stats")
    print(f"   结果: {result.output[:100]}...")
    
    print("\n[完成] 记忆工具测试完成")


def test_rag_tool():
    """测试RAG工具"""
    print("\n=== 测试RAG工具 ===")
    
    tool = RAGTool()
    
    # 测试摄取文档
    print("\n1. 测试摄取文档...")
    result = tool.run(
        operation="ingest",
        content="""
        AI代理系统是一种能够自主执行任务的软件系统。
        它通常包括感知、决策和执行三个主要模块。
        现代AI代理可以处理自然语言、图像和结构化数据。
        """,
        pipeline_type="simple"
    )
    print(f"   结果: {result.output}")
    print(f"   成功: {result.success}")
    
    # 测试查询知识
    print("\n2. 测试查询知识...")
    result = tool.run(
        operation="query",
        content="AI代理系统有哪些主要模块？",
        pipeline_type="simple",
        top_k=2
    )
    print(f"   结果: {result.output[:200]}...")
    print(f"   成功: {result.success}")
    
    # 测试清空
    print("\n3. 测试清空知识库...")
    result = tool.run(
        operation="clear",
        pipeline_type="simple"
    )
    print(f"   结果: {result.output}")
    
    print("\n[完成] RAG工具测试完成")


def test_integration():
    """测试集成"""
    print("\n=== 测试集成功能 ===")
    
    from src.memory import WorkingMemory, EpisodicMemory
    from src.memory.storage import DocumentStore
    
    # 测试工作记忆
    print("\n1. 测试工作记忆...")
    working_mem = WorkingMemory()
    
    # 创建MemoryItem对象
    from src.memory.base import MemoryItem
    import uuid
    from datetime import datetime
    
    memory_item = MemoryItem(
        id=str(uuid.uuid4()),
        content="临时记住用户的电话号码：13800138000",
        memory_type=MemoryType.WORKING,
        metadata={"ttl": 60}
    )
    
    item_id = working_mem.store(memory_item)
    print(f"   工作记忆存储成功，ID: {item_id}")
    
    # 测试文档存储
    print("\n2. 测试文档存储...")
    doc_store = DocumentStore()
    success = doc_store.store_document(
        doc_id="test_doc_001",
        content="这是一个测试文档，用于验证文档存储功能。",
        doc_type="test",
        metadata={"author": "tester", "version": "1.0"}
    )
    print(f"   文档存储成功: {success}")
    
    print("\n[完成] 集成测试完成")


if __name__ == "__main__":
    print("开始测试记忆子系统...")
    print("=" * 50)
    
    try:
        test_memory_manager()
        test_memory_tool()
        test_rag_tool()
        test_integration()
        
        print("\n" + "=" * 50)
        print("[完成] 所有测试完成！")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()