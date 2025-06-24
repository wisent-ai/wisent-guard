#!/usr/bin/env python3
"""
Main entry point for running the Wisent-Guard MCP server.
"""

import asyncio
import sys
import argparse
import logging

from .server import run_server, WisentGuardMCPServer


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Wisent-Guard MCP Server for model self-reflection"
    )
    
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name to use for analysis"
    )
    
    parser.add_argument(
        "--default-layer",
        type=int,
        default=15,
        help="Default layer for steering operations"
    )
    
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        default=True,
        help="Enable performance tracking"
    )
    
    parser.add_argument(
        "--disable-tracking",
        action="store_true",
        help="Disable performance tracking"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode instead of MCP server"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    if args.demo:
        # Run demo mode
        logger.info("Running in demo mode")
        from .demo import run_demo
        await run_demo()
        return
    
    # Determine tracking setting
    enable_tracking = args.enable_tracking and not args.disable_tracking
    
    logger.info(f"Starting Wisent-Guard MCP Server")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Default layer: {args.default_layer}")
    logger.info(f"Tracking enabled: {enable_tracking}")
    
    try:
        # Check if MCP is available
        try:
            import mcp
        except ImportError:
            logger.error("MCP package not available. Install with: pip install mcp")
            sys.exit(1)
        
        # Run the server
        await run_server()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 