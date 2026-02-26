"""
Log collector for capturing logs from Debuggable objects and building execution graphs.
"""
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict


class LogCollector:
    """
    Collects logs from Debuggable objects and builds execution graph structure.

    This collector captures logs with their parent-child relationships,
    organizing them by debuggable ID for visualization and debugging.

    Attributes:
        logs (List[Dict]): All collected logs in order
        log_groups (Dict[str, List[Dict]]): Logs organized by debuggable ID
        graph_nodes (Dict[str, Dict]): Graph nodes indexed by debuggable ID
        graph_edges (Set[Tuple[str, str]]): Set of (parent_id, child_id) edges
    """

    def __init__(self):
        """Initialize the log collector."""
        self.logs: List[Dict[str, Any]] = []
        self.log_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.graph_nodes: Dict[str, Dict[str, Any]] = {}
        self.graph_edges: Set[Tuple[str, str]] = set()
        self._node_ids: Set[str] = set()

    def __call__(self, log_data: Dict[str, Any]):
        """
        Collect a log entry.

        Args:
            log_data: Log data dictionary from Debuggable.log()
        """
        # Add timestamp if not present
        if 'timestamp' not in log_data:
            from datetime import datetime
            log_data['timestamp'] = datetime.now().isoformat()

        # Store the log
        self.logs.append(log_data)

        # Organize by debuggable ID
        node_id = log_data.get('id', 'unknown')
        self.log_groups[node_id].append(log_data)

        # Track this node ID
        self._node_ids.add(node_id)

        # Build graph structure
        self._update_graph(log_data)

    def _update_graph(self, log_data: Dict[str, Any]):
        """
        Update the graph structure based on log data.

        Extracts parent-child relationships from parent_ids field.
        ALL entries in parent_ids are valid parents.

        Args:
            log_data: Log data with parent_ids information
        """
        node_id = log_data.get('id', 'unknown')
        node_name = log_data.get('name', node_id)
        log_type = log_data.get('type', '')

        # Create or update node
        if node_id not in self.graph_nodes:
            self.graph_nodes[node_id] = {
                'id': node_id,
                'name': node_name,
                'label': node_name,  # Keep for compatibility
                'log_count': 0,
                'node_type': 'normal'  # Track node type
            }
        else:
            # Update name if this is a real log entry (not just a placeholder from parent_ids)
            # This ensures we use the actual log name instead of the derived placeholder name
            self.graph_nodes[node_id]['name'] = node_name
            self.graph_nodes[node_id]['label'] = node_name

        self.graph_nodes[node_id]['log_count'] += 1

        # Extract parent-child relationships
        # ALL entries in parent_ids are valid parents, not just one
        parent_ids = log_data.get('parent_ids', [])
        for parent_id in parent_ids:
            # Add edge (parent -> child), set automatically handles duplicates
            self.graph_edges.add((parent_id, node_id))

            # Ensure parent node exists (create placeholder if needed)
            if parent_id not in self.graph_nodes:
                # Try to derive a reasonable name from the ID
                # Handle both hyphen and underscore separators
                if '-' in parent_id:
                    parent_name = parent_id.split('-')[0]
                elif '_' in parent_id:
                    parent_name = parent_id.split('_')[0]
                else:
                    parent_name = parent_id

                self.graph_nodes[parent_id] = {
                    'id': parent_id,
                    'name': parent_name,
                    'label': parent_name,  # Keep for compatibility
                    'log_count': 0,
                    'node_type': 'normal'
                }

        # Handle AgentWorkstreamCompleted - create exit node
        if log_type == 'AgentWorkstreamCompleted':
            # Create a symbolic exit node
            exit_node_id = f"{node_id}-exit"
            exit_node_name = "Exit"

            if exit_node_id not in self.graph_nodes:
                self.graph_nodes[exit_node_id] = {
                    'id': exit_node_id,
                    'name': exit_node_name,
                    'label': exit_node_name,
                    'log_count': 0,
                    'node_type': 'exit'  # Mark as exit node
                }

            # Create edge from current node to exit node
            self.graph_edges.add((node_id, exit_node_id))

    def get_graph_structure(self) -> Dict[str, Any]:
        """
        Get the complete graph structure.

        Returns:
            Dictionary with 'nodes', 'edges', and 'agent' keys:
            - nodes: List of node dictionaries
            - edges: List of edge dictionaries with source/target
            - agent: Root agent node (first root with 'Agent' in ID, or first root)
        """
        # Convert nodes dict to list
        nodes = list(self.graph_nodes.values())

        # Convert edges set to list of dicts
        edges = [{'source': src, 'target': tgt} for src, tgt in self.graph_edges]

        # Find root nodes (nodes with no incoming edges)
        nodes_with_parents = {edge['target'] for edge in edges}
        root_nodes = [node for node in nodes if node['id'] not in nodes_with_parents]

        # Find agent node (prefer root with 'Agent' in ID)
        agent_node = None
        for node in root_nodes:
            if 'Agent' in node['id']:
                agent_node = {
                    'id': node['id'],
                    'name': node['name'],
                    'log_count': node['log_count']
                }
                break

        # Fallback to first root if no agent found
        if not agent_node and root_nodes:
            node = root_nodes[0]
            agent_node = {
                'id': node['id'],
                'name': node['name'],
                'log_count': node['log_count']
            }

        # Default agent if none found
        if not agent_node:
            agent_node = {'id': 'Agent_default', 'name': 'Agent', 'log_count': 0}

        return {
            'nodes': nodes,
            'edges': edges,
            'agent': agent_node
        }

    def get_logs_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all logs for a specific debuggable node.

        Args:
            node_id: The debuggable ID

        Returns:
            List of log entries for that node
        """
        return self.log_groups.get(node_id, [])

    def get_all_node_ids(self) -> List[str]:
        """
        Get all debuggable node IDs.

        Returns:
            List of all node IDs
        """
        return sorted(self._node_ids)

    def clear(self):
        """Clear all collected logs and graph data."""
        self.logs.clear()
        self.log_groups.clear()
        self.graph_nodes.clear()
        self.graph_edges.clear()
        self._node_ids.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert collected logs to a dictionary for storage.

        Returns:
            Dictionary with logs and metadata
        """
        return {
            'logs': self.logs,
            'log_groups': dict(self.log_groups),
            'graph_nodes': self.graph_nodes,
            'graph_edges': list(self.graph_edges),
            'graph_structure': self.get_graph_structure()
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'LogCollector':
        """
        Create a LogCollector from a dictionary.

        Args:
            data: Dictionary with logs and metadata

        Returns:
            LogCollector instance
        """
        collector = LogCollector()
        collector.logs = data.get('logs', [])
        collector.log_groups = defaultdict(list, data.get('log_groups', {}))
        collector.graph_nodes = data.get('graph_nodes', {})
        collector.graph_edges = set(tuple(edge) if isinstance(edge, list) else edge 
                                     for edge in data.get('graph_edges', []))
        collector._node_ids = set(collector.log_groups.keys())
        return collector

    @staticmethod
    def from_json_logs(log_path: str, json_file_pattern:str ='*') -> 'LogCollector':
        """
        Create a LogCollector by reading logs from JSON files.

        Handles both single file and folder of JSON files.
        Uses iter_json_objs which automatically handles both cases.

        Args:
            log_path: Path to a JSON file or folder containing JSON files
            json_file_pattern (str): File pattern to use when searching for JSON files in the log directory.

        Returns:
            LogCollector with all logs loaded and graph built

        Examples:
            >>> collector = LogCollector.from_json_logs("logs/agent_execution/")
            >>> len(collector.logs) > 0
            True
        """
        from rich_python_utils.io_utils.json_io import iter_json_objs
        
        collector = LogCollector()
        
        try:
            # iter_json_objs handles both files and directories automatically
            log_entries = list(iter_json_objs(
                log_path,
                use_tqdm=False,
                verbose=False,
                json_file_pattern=json_file_pattern
            ))
            
            # Add each log entry - collector will build the graph automatically
            for log_entry in log_entries:
                collector(log_entry)
                
        except Exception as e:
            print(f"[WARNING] Failed to read logs from {log_path}: {e}")
        
        return collector

    def get_log_graph_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about the log graph structure.

        Returns:
            Dictionary with node_count, edge_count, max_depth, total_log_count, etc.

        Examples:
            >>> collector = LogCollector()
            >>> collector({'id': 'A', 'name': 'Agent', 'item': 'test'})
            >>> collector({'id': 'B', 'name': 'Node', 'item': 'test', 'parent_ids': ['A']})
            >>> stats = collector.get_log_graph_statistics()
            >>> stats['node_count']
            2
            >>> stats['edge_count']
            1
        """
        graph_structure = self.get_graph_structure()
        nodes = graph_structure.get('nodes', [])
        edges = graph_structure.get('edges', [])
        agent = graph_structure.get('agent', {})

        # Calculate total log count
        total_log_count = sum(node.get('log_count', 0) for node in nodes)

        # Calculate max depth (simple BFS-based depth calculation)
        max_depth = 0
        if nodes and edges:
            # Build adjacency list
            children = {}
            for edge in edges:
                src = edge['source']
                if src not in children:
                    children[src] = []
                children[src].append(edge['target'])

            # BFS from agent node
            agent_id = agent.get('id')
            if agent_id:
                queue = [(agent_id, 0)]
                visited = set()

                while queue:
                    node_id, depth = queue.pop(0)
                    if node_id in visited:
                        continue
                    visited.add(node_id)
                    max_depth = max(max_depth, depth)

                    # Add children
                    for child in children.get(node_id, []):
                        queue.append((child, depth + 1))

        return {
            'node_count': len(nodes),
            'edge_count': len(edges),
            'total_log_count': total_log_count,
            'max_depth': max_depth,
            'agent_id': agent.get('id', 'unknown'),
            'agent_name': agent.get('name', 'unknown')
        }
