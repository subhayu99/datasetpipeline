import tomllib
import re

def extract_package_name(dep_string):
    """Extract package name from dependency string, handling various formats."""
    # Handle cases like "torch[cuda]", "package>=1.0", "package==1.0", etc.
    name = re.split(r'[>=<!=~\[]', dep_string)[0].strip()
    return name.lower()

def update_dependency_list(deps, versions):
    """Update a list of dependencies with pinned versions."""
    updated_deps = []
    changes = []
    
    for dep in deps:
        name = extract_package_name(dep)
        if '==' not in dep and name in versions:
            # Pin the version
            if '[' in dep:
                # Handle extras like "torch[cuda]"
                base_name, extras = dep.split('[', 1)
                new_dep = f"{base_name.strip()}=={versions[name]}[{extras}"
            else:
                new_dep = f"{name}=={versions[name]}"
            updated_deps.append(new_dep)
            changes.append((dep, new_dep))
        else:
            updated_deps.append(dep)
    
    return updated_deps, changes

def main():
    # Read the lock file
    try:
        with open('uv.lock', 'rb') as f:
            lock_data = tomllib.load(f)
    except FileNotFoundError:
        print("❌ Error: uv.lock file not found!")
        return
    except Exception as e:
        print(f"❌ Error reading uv.lock: {e}")
        return

    # Read pyproject.toml as text to preserve formatting
    try:
        with open('pyproject.toml', 'r', encoding='utf-8') as f:
            toml_content = f.read()
    except FileNotFoundError:
        print("❌ Error: pyproject.toml file not found!")
        return
    except Exception as e:
        print(f"❌ Error reading pyproject.toml: {e}")
        return

    # Also read as parsed TOML for analysis
    try:
        pyproject = tomllib.loads(toml_content)
    except Exception as e:
        print(f"❌ Error parsing pyproject.toml: {e}")
        return

    # Extract versions from lock file
    versions = {}
    for package in lock_data.get('package', []):
        name = package['name'].lower()
        versions[name] = package['version']

    print(f"📦 Found {len(versions)} packages in uv.lock")
    print("=" * 60)

    # Track all changes
    all_changes = []
    updated_content = toml_content

    # Process main dependencies
    current_deps = pyproject.get('project', {}).get('dependencies', [])
    if current_deps:
        print("\n🔧 UPDATING MAIN DEPENDENCIES:")
        print("-" * 40)
        
        updated_deps, changes = update_dependency_list(current_deps, versions)
        
        if changes:
            # Replace the dependencies array in the file
            # Find the dependencies section and replace it
            lines = updated_content.split('\n')
            in_deps = False
            deps_start = -1
            deps_end = -1
            bracket_count = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('dependencies = ['):
                    in_deps = True
                    deps_start = i
                    if line.strip().endswith(']'):
                        deps_end = i
                        break
                    bracket_count = 1
                elif in_deps:
                    bracket_count += line.count('[') - line.count(']')
                    if bracket_count == 0:
                        deps_end = i
                        break
            
            if deps_start >= 0 and deps_end >= 0:
                # Build new dependencies section
                new_deps_lines = ['dependencies = [']
                for dep in updated_deps:
                    new_deps_lines.append(f'    "{dep}",')
                new_deps_lines.append(']')
                
                # Replace in content
                new_lines = lines[:deps_start] + new_deps_lines + lines[deps_end + 1:]
                updated_content = '\n'.join(new_lines)
                
                for old_dep, new_dep in changes:
                    print(f"  ✅ {old_dep} → {new_dep}")
                    all_changes.append((old_dep, new_dep))
        else:
            print("  ✓ All main dependencies already pinned or not in lock file")

    # Process optional dependencies
    optional_deps = pyproject.get('project', {}).get('optional-dependencies', {})
    if optional_deps:
        print("\n🔧 UPDATING OPTIONAL DEPENDENCIES:")
        print("-" * 40)
        
        for group_name, deps in optional_deps.items():
            print(f"\n[{group_name}]:")
            updated_deps, changes = update_dependency_list(deps, versions)
            
            if changes:
                # Find and replace this optional dependency group
                group_pattern = f'{group_name} = ['
                lines = updated_content.split('\n')
                
                for i, line in enumerate(lines):
                    if group_pattern in line:
                        # Find the end of this array
                        start_line = i
                        bracket_count = line.count('[') - line.count(']')
                        end_line = i
                        
                        if bracket_count > 0:
                            for j in range(i + 1, len(lines)):
                                bracket_count += lines[j].count('[') - lines[j].count(']')
                                if bracket_count == 0:
                                    end_line = j
                                    break
                        
                        # Build new group
                        indent = '    ' if 'project.optional-dependencies' in updated_content else ''
                        new_group_lines = [f'{indent}{group_name} = [']
                        for dep in updated_deps:
                            new_group_lines.append(f'{indent}    "{dep}",')
                        new_group_lines.append(f'{indent}]')
                        
                        # Replace in content
                        new_lines = lines[:start_line] + new_group_lines + lines[end_line + 1:]
                        updated_content = '\n'.join(new_lines)
                        break
                
                for old_dep, new_dep in changes:
                    print(f"    ✅ {old_dep} → {new_dep}")
                    all_changes.append((old_dep, new_dep))
            else:
                print(f"    ✓ All dependencies in [{group_name}] already pinned or not in lock file")

    # Write the updated content back
    if all_changes:
        try:
            # Create backup
            with open('pyproject.toml.backup', 'w', encoding='utf-8') as f:
                f.write(toml_content)
            print("\n💾 Created backup: pyproject.toml.backup")
            
            # Write updated file
            with open('pyproject.toml', 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"\n✅ SUCCESS! Updated pyproject.toml with {len(all_changes)} pinned dependencies:")
            for old_dep, new_dep in all_changes:
                print(f"  • {old_dep} → {new_dep}")
                
        except Exception as e:
            print(f"\n❌ Error writing updated pyproject.toml: {e}")
            return
    else:
        print("\n✅ No changes needed - all dependencies are already pinned!")

    print("\n📊 SUMMARY:")
    print("-" * 40)
    print(f"Total packages in lock file: {len(versions)}")
    print(f"Dependencies updated: {len(all_changes)}")
    if all_changes:
        print("Backup created: pyproject.toml.backup")

if __name__ == "__main__":
    main()