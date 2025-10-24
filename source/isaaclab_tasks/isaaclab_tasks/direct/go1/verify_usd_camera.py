"""
Verify that the USD file has the belly camera properly configured
Run this to check the USD structure before using it in the environment
"""

from pxr import Usd, UsdGeom, Sdf
import sys


def verify_usd_camera():
    """Check USD file for camera configuration"""

    usd_path = "/home/sripu715/Downloads/go1_cams_2.usd"

    print("\n" + "=" * 80)
    print("🔍 USD Camera Verification")
    print("=" * 80)
    print(f"Checking: {usd_path}\n")

    try:
        # Open USD stage with load all payloads
        stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
        if not stage:
            print(f"❌ ERROR: Could not open USD file at {usd_path}")
            return False

        print("✅ USD file loaded successfully\n")

        # Print all prims with more detail
        print("📋 Full USD Structure:")
        for prim in stage.Traverse():
            indent = "   " * (len(str(prim.GetPath()).split('/')) - 2)
            prim_type = prim.GetTypeName()
            is_camera = "📷" if UsdGeom.Camera(prim) else ""
            print(f"{indent}- {prim.GetPath()} [{prim_type}] {is_camera}")

        print("\n" + "-" * 80 + "\n")

        # Look for go1 prim (the payload reference)
        go1_prim = stage.GetPrimAtPath("/World/go1")
        if not go1_prim or not go1_prim.IsValid():
            print("❌ ERROR: /World/go1 prim not found or invalid")
            return False

        print(f"✅ Found Go1 root at: /World/go1")

        # Check if payload is loaded
        if go1_prim.HasPayload():
            print(f"⚠️  Go1 has payload reference (external USD file)")
            payloads = go1_prim.GetMetadata('payload')
            if payloads:
                print(f"   Payload: {payloads}")

        # Look for trunk in children of go1
        trunk_path = None

        def find_trunk(prim, depth=0):
            """Recursively find trunk prim"""
            if depth > 10:  # Prevent infinite recursion
                return None

            if "trunk" in prim.GetName().lower() or prim.GetName() == "trunk":
                return prim.GetPath()

            for child in prim.GetAllChildren():
                result = find_trunk(child, depth + 1)
                if result:
                    return result
            return None

        trunk_path = find_trunk(go1_prim)

        if not trunk_path:
            print("\n❌ ERROR: Could not find 'trunk' prim under /World/go1")
            print("\n📋 Children of /World/go1:")
            for child in go1_prim.GetAllChildren():
                print(f"   - {child.GetPath()} [{child.GetTypeName()}]")
            return False

        print(f"✅ Found trunk at: {trunk_path}")

        # Look for camera under trunk
        camera_found = False
        camera_path = None

        trunk_prim = stage.GetPrimAtPath(trunk_path)

        def find_cameras(prim, depth=0):
            """Recursively find all cameras"""
            cameras = []
            if depth > 5:
                return cameras

            if UsdGeom.Camera(prim):
                cameras.append(prim)

            for child in prim.GetAllChildren():
                cameras.extend(find_cameras(child, depth + 1))

            return cameras

        cameras = find_cameras(trunk_prim)

        if not cameras:
            print(f"\n❌ ERROR: No camera found under trunk at {trunk_path}")
            print(f"\n📋 Children of trunk:")
            for child in trunk_prim.GetAllChildren():
                print(f"   - {child.GetPath()} (Type: {child.GetTypeName()})")

            print(f"\n💡 To add a camera in Isaac Sim:")
            print(f"   1. Open: {usd_path}")
            print(f"   2. Navigate to: {trunk_path}")
            print(f"   3. Right-click trunk → Create → Camera")
            print(f"   4. Name it 'belly_cam'")
            print(f"   5. Set position: (0, 0, -0.01)")
            print(f"   6. Set rotation to point down")
            print(f"   7. Save USD file")
            return False

        # Process found cameras
        for camera_prim in cameras:
            camera_path = camera_prim.GetPath()
            print(f"✅ Found camera at: {camera_path}")

            camera = UsdGeom.Camera(camera_prim)
            xform = UsdGeom.Xformable(camera_prim)

            print(f"\n📷 Camera Properties:")
            print(f"   Name: {camera_prim.GetName()}")
            print(f"   Full Path: {camera_path}")
            print(f"   Parent: {camera_prim.GetParent().GetPath()}")

            # Get camera attributes
            focal_length = camera.GetFocalLengthAttr().Get()
            h_aperture = camera.GetHorizontalApertureAttr().Get()
            v_aperture = camera.GetVerticalApertureAttr().Get()

            if focal_length:
                print(f"   Focal Length: {focal_length}mm")
            if h_aperture:
                print(f"   Horizontal Aperture: {h_aperture}mm")
            if v_aperture:
                print(f"   Vertical Aperture: {v_aperture}mm")

            # Get local transform
            local_transform = xform.GetLocalTransformation()
            print(f"\n🔧 Local Transform Matrix:")
            for i in range(4):
                row = local_transform.GetRow(i)
                print(f"   [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}, {row[3]:8.4f}]")

            # Extract position
            translation = local_transform.ExtractTranslation()
            print(
                f"\n📍 Position (relative to trunk): [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")

            camera_found = True

        if not camera_found:
            return False

        # Success summary
        print(f"\n✅ Verification Complete!")
        print(f"\n📋 Summary:")
        print(f"   Go1 Root: /World/go1")
        print(f"   Trunk: {trunk_path}")
        print(f"   Camera: {camera_path}")
        print(f"   Hierarchy: ✅ Camera is child of trunk")

        # Extract relative path from /World/go1
        relative_camera_path = str(camera_path).replace("/World/go1", "")

        print(f"\n💡 Usage in IsaacLab Config:")
        print(f"   Camera prim_path should be:")
        print(f"   '{{ENV_REGEX_NS}}/Robot{relative_camera_path}'")
        print(f"\n   Example:")
        print(f"   camera = CameraCfg(")
        print(f"       prim_path='{{ENV_REGEX_NS}}/Robot{relative_camera_path}',")
        print(f"       ...\n   )")

        print(f"\n🎯 The GroundPlane in USD is fine - it's just scene decoration")
        print(f"   IsaacLab will use its own terrain system")

        print("\n" + "=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_usd_camera()
    sys.exit(0 if success else 1)