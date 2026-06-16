#!/usr/bin/env python3
"""
Hermes Parallel Agents Framework
基于官方文档实现的并行委派框架
"""
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from src.agent.parent_agent import ParentAgent
from src.core.interrupt import interrupt_manager

console = Console()

def print_banner():
    console.print(Panel(
        "[bold cyan]⚡ Hermes Parallel Agents Framework[/bold cyan]\n\n"
        "[white]核心特性:[/white]\n"
        "  [green]✓[/green] 子 Agent 完全隔离上下文\n"
        "  [green]✓[/green] 最多 3 个子 Agent 并行\n"
        "  [green]✓[/green] 禁用工具集严格管控\n"
        "  [green]✓[/green] 中断信号传播\n"
        "  [green]✓[/green] 仅摘要进入父 Agent 上下文\n\n"
        "[dim]命令: quit / reset / help / Ctrl+C 中断[/dim]",
        border_style="cyan", width=58
    ))

def print_help():
    table = Table(title="示例指令", box=box.ROUNDED)
    table.add_column("类型", style="cyan")
    table.add_column("示例", style="yellow")
    
    table.add_row("单任务委派", "帮我检查 workspace/app.py 是否有语法错误")
    table.add_row("并行研究",   "并行研究：Python 3.13 新特性 + Rust 2024 edition + WebAssembly 现状")
    table.add_row("代码审查",   "审查 workspace/src 目录的安全问题并修复")
    table.add_row("直接问答",   "什么是 ReAct 框架？")
    
    console.print(table)

def main():
    print_banner()
    
    # 初始化工作目录
    Path("./workspace").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    agent = ParentAgent()
    
    console.print("\n[green]✅ 父 Agent (深度 0) 就绪[/green]")
    console.print("[dim]输入 'help' 查看示例，Ctrl+C 触发中断传播[/dim]\n")
    
    while True:
        try:
            user_input = console.input("[bold yellow]You:[/bold yellow] ").strip()
            if not user_input:
                continue
            
            match user_input.lower():
                case "quit" | "exit":
                    console.print("[dim]👋 再见！[/dim]")
                    break
                case "reset":
                    agent.reset()
                    console.print("[green]🔄 已重置[/green]")
                    continue
                case "help":
                    print_help()
                    continue
            
            with console.status("[bold green]🤖 Agent 处理中...[/bold green]"):
                response = agent.run(user_input)
            
            console.print(f"\n[bold blue]Hermes:[/bold blue]")
            console.print(Markdown(response))
            console.print()
        
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ 中断信号 → 传播至所有子 Agent[/yellow]")
            interrupt_manager.interrupt_all()
        except Exception as e:
            console.print(f"[red]❌ 错误: {e}[/red]")

if __name__ == "__main__":
    main()