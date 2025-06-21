#!/usr/bin/env python3
"""
Debug script to check pipeline module
"""
import sys
import os

def debug_pipeline():
    """Debug the pipeline module imports"""
    print("🔍 Debugging Pipeline Module")
    print("=" * 50)
    
    # Check if pipeline.py exists
    if os.path.exists("pipeline.py"):
        print("✅ pipeline.py file exists")
        
        # Check file size
        file_size = os.path.getsize("pipeline.py")
        print(f"📄 File size: {file_size} bytes")
        
        # Check if file is readable
        try:
            with open("pipeline.py", "r") as f:
                content = f.read(1000)  # Read first 1000 chars
                print(f"📖 File is readable, starts with: {content[:100]}...")
        except Exception as e:
            print(f"❌ Cannot read file: {e}")
            return False
    else:
        print("❌ pipeline.py file does not exist")
        return False
    
    # Try to import the module
    try:
        print("\n🔄 Attempting to import pipeline module...")
        import pipeline
        print("✅ Successfully imported pipeline module")
        
        # Check what's available in the module
        available_classes = []
        available_functions = []
        
        for name in dir(pipeline):
            if not name.startswith('_'):
                obj = getattr(pipeline, name)
                if isinstance(obj, type):
                    available_classes.append(name)
                elif callable(obj):
                    available_functions.append(name)
        
        print(f"\n📋 Available classes: {available_classes}")
        print(f"📋 Available functions: {available_functions}")
        
        # Check for specific classes
        target_classes = ['EnhancedRAGPipeline', 'RAGPipeline']
        for class_name in target_classes:
            if hasattr(pipeline, class_name):
                print(f"✅ Found class: {class_name}")
                try:
                    cls = getattr(pipeline, class_name)
                    instance = cls()
                    print(f"✅ Successfully created instance of {class_name}")
                    return True
                except Exception as e:
                    print(f"❌ Error creating instance of {class_name}: {e}")
            else:
                print(f"❌ Class not found: {class_name}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        
        # Check for syntax errors
        try:
            with open("pipeline.py", "r") as f:
                code = f.read()
            compile(code, "pipeline.py", "exec")
            print("✅ No syntax errors found")
        except SyntaxError as se:
            print(f"❌ Syntax error in pipeline.py: {se}")
            print(f"   Line {se.lineno}: {se.text}")
        except Exception as ce:
            print(f"❌ Compilation error: {ce}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    return False

def fix_pipeline_imports():
    """Try to fix common import issues"""
    print("\n🔧 Attempting to fix import issues...")
    
    # Clear Python cache
    import glob
    cache_files = glob.glob("__pycache__/*") + glob.glob("*.pyc")
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            print(f"🗑️ Removed cache file: {cache_file}")
        except:
            pass
    
    # Try to reload modules
    modules_to_reload = ['config', 'retriever', 'generator', 'query_enhancer', 'pipeline']
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                import importlib
                importlib.reload(sys.modules[module_name])
                print(f"🔄 Reloaded module: {module_name}")
            except Exception as e:
                print(f"⚠️ Could not reload {module_name}: {e}")

if __name__ == "__main__":
    success = debug_pipeline()
    
    if not success:
        print("\n" + "=" * 50)
        print("🔧 ATTEMPTING FIXES")
        print("=" * 50)
        fix_pipeline_imports()
        
        print("\n" + "=" * 50)
        print("🔄 RETRYING AFTER FIXES")
        print("=" * 50)
        success = debug_pipeline()
    
    if success:
        print("\n✅ Pipeline debugging successful!")
    else:
        print("\n❌ Pipeline debugging failed!")
        print("\nSuggested fixes:")
        print("1. Check that pipeline.py contains the complete updated code")
        print("2. Restart your Python environment")
        print("3. Clear all cache files")
        print("4. Check for circular import issues")