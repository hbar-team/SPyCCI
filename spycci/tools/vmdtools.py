import sh, os, shutil, logging
import imageio.v2 as imageio
import numpy as np

from copy import deepcopy
from tempfile import NamedTemporaryFile as tmp
from tempfile import mkdtemp

from os import system
from os.path import join, basename, isfile
from typing import List, Optional, Union

from spycci.core.dependency_finder import locate_vmd
from spycci.core.spectroscopy import VibrationalData
from spycci.tools.cubetools import Cube
from spycci.systems import System, ReactionPath

import logging
logger = logging.getLogger(__name__)


class VMDRenderer:
    """
    The `VMDRenderer` class is a simple wrapper developed around the Visual Molecular Dynamics (VMD)
    software. The class allows the user to easily generate images and renders of molecules and cube files.
    Once an instance of the `VMDRenderer` is created, the user can use its render functionality through
    the provided built-in methods.

    Arguments
    ---------
    resolution: Union[int, List[int]]
        The resolution of the output image. This argument accepts either a single integer, which sets the
        same resolution for both the X and Y axes (producing a square image), or a list of two integers,
        which independently specify the resolution along the X (width) and Y (height) axes (producing a
        rectangular image). (default: 800).
    scale: float
        Scales the frame zoom according to the user specified factor (default: 1. no
        zoom is applied)
    xyz_translation: List[float]
        A list of three float values defining the translation vector (X, Y, Z) to apply to the
        object. (default: [0., 0., 0.] no translation is applied).
    xyx_rotation: List[float]
        The list of 3 rotation angles (from 0 to 360°) defyning subsequent rotations around
        the X, Y and X axis. (default: [0., 0., 0.] no rotation is applied).
    shadows: bool
        If set to `True` will enable the vmd shadows option. (default: True)
    ambientocclusion: bool
        If set to `True` will enable the vmd ambientocclusion option. (default: True)
    dof: bool
        If set to `True` will enable the vmd dof option. (default: True)
    show_axes: bool
        If set to `True` will show the axes representation in the render window. (default: False)
    suppress_output: bool
        If set to `True` will run the rendering without printing any message from `vmd`. (default: False)
    VMD_PATH: str
        The path to the vmd executable. Is set to `None` (default), will automatically search `vmd`
        in the system PATH.

    Raises
    ------
    FileNotFoundError
        Exception raised if the user provided `VMD_PATH` is invalid.
    RuntimeError
        Exception raised if either the `vmd` program or the Tachyon ray tracer are not found.
    ValueError
        Exception raised if scale or xyx rotation angles are not properly formatted.
    """

    def __init__(
        self,
        resolution: Union[int, List[int]] = 800,
        scale: float = 1.0,
        xyz_translation: List[float] = [0.0, 0.0, 0.0],
        xyx_rotation: List[float] = [0.0, 0.0, 0.0],
        shadows: bool = True,
        ambientocclusion: bool = True,
        dof: bool = True,
        show_axes: bool = False,
        suppress_output: bool = False,
        VMD_PATH: Optional[str] = None,
    ) -> None:
        
        # Store basic rendering settings
        self.shadows: bool = shadows
        self.ambientocclusion: bool = ambientocclusion
        self.dof: bool = dof
        self.show_axes: bool = show_axes
        self.suppress_output: bool = suppress_output

        # Define the attributes to be set using properties
        self.__scale: float = None
        self.__resolution: List[int] = None
        self.__xyz_translation: List[float] = None
        self.__xyx_rotation: List[float] = None
        
        # Set the protected attributes using propery setters
        self.scale = scale
        self.resolution = resolution
        self.xyz_translation = xyz_translation
        self.xyx_rotation = xyx_rotation
        
        # Search for the VMD folder and set the vmd root variable
        self.__vmd_root = None
        if VMD_PATH and isfile(VMD_PATH) is False:
            raise FileNotFoundError(
                f"The VMD_PATH {VMD_PATH} does not point to a valid executable."
            )

        elif VMD_PATH:
            self.__vmd_root = locate_vmd(VMD_PATH).removesuffix("/bin/vmd")

        else:
            self.__vmd_root = locate_vmd().removesuffix("/bin/vmd")

        # Check the availability of the tachyon ray tracer
        self.__tachyon_path = join(self.__vmd_root, "lib/vmd/tachyon_LINUXAMD64")
        if isfile(self.__tachyon_path) is False:
            raise RuntimeError(
                f"Cannot locate the Tachyon ray tracer (required by VMD)."
            )

    @property
    def scale(self) -> float:
        """
        The float value setting the frame zoom. The value 1. is set by VMD on startup
        to render the whole molecule in the unrotated frame.

        Return
        ------
        float
            The scale of the frame.
        """
        return self.__scale

    @scale.setter
    def scale(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(
                "The VMD frame scale factor must be a non-zero positive float."
            )
        self.__scale: float = value

    @property
    def xyz_translation(self) -> List[float]:
        """
        The list of three float values defining the translation vector (X, Y, Z) to apply to the
        object.

        Return
        ------
        List[float]
            The translation vector.
        """
        return self.__xyz_translation

    @xyz_translation.setter
    def xyz_translation(self, value: List[float]) -> None:
        if len(value) != 3:
            raise ValueError("The translation vector must be a list of 3 float values.")
        self.__xyz_translation: float = [float(v) for v in value]
    

    @property
    def xyx_rotation(self) -> List[float]:
        """
        The list of 3 rotation angles (from 0 to 360°) defyning subsequent rotations around
        the X, Y and X axis.

        Return
        ------
        List[float]
            The list of XYX rotation angles.
        """
        return self.__xyx_rotation

    @xyx_rotation.setter
    def xyx_rotation(self, value: List[float]) -> None:
        if len(value) != 3:
            raise ValueError("The XYX rotation must be a list of 3 float values.")
        self.__xyx_rotation: float = [float(v) for v in value]

    
    @property
    def resolution(self) -> List[int]:
        """
        The list of two integer values setting the XY resolution of the output image.

        Returns
        -------
        List[int]
            The list encoding the image resolution across the X and Y axes.
        """
        return self.__resolution
    
    @resolution.setter
    def resolution(self, value: Union[int, List[int]]) -> None:

        if isinstance(value, (list, tuple)) and len(value) == 2:
            self.__resolution = [int(r) for r in value]
        
        elif isinstance(value, int):
            self.__resolution = [value, value]
        
        else:
            raise ValueError("Resolution must be either an int or a list of two integers.")
        

    ############################
    # Core rendering functions #
    ############################

    def render_system_file(
        self,
        molecule_file: str,
        filename: Optional[str] = None,
    ) -> None:
        """
        Given the path to a molecule file (e.g. .xyz, .pdb) the function saves a `.bmp` render
        of the molecular structure.

        Arguments
        ---------
        molecule_file: str
            The path to the file encoding the structure of the molecule.
        filename: Optional[str]
            The name, or the path, of the output `.bmp` file. If `None` (default) the output file
            will be generated from the root of the input filename (e.g. `root.bmp` from `root.xyz`).
        """
        # Check if given file exists
        if isfile(molecule_file) is False:
            raise FileNotFoundError(f"Unable to find the {molecule_file} file.")

        root_name = basename(molecule_file).rsplit(".", 1)[0]
        script = self._tcl_script_preamble()

        # Load the molecule from file and plot its backbone
        script += f"mol new {molecule_file}\n"
        script += self._tcl_plot_backbone()

        filename = filename.removesuffix(".bmp") if filename is not None else root_name
        self._render(script, filename)

    def render_cube_file(
        self,
        cubefile: str,
        isovalue: Optional[float] = None,
        positive_color: int = 1,
        negative_color: int = 0,
        show_negative: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """
        Given the path to a generic `.cube` file, saves a `.bmp` render of the contained
        volumetric data.

        Arguments
        ---------
        cubefile: str
            The path to the `.fukui.cube` file that must be rendered.
        isovalue: Optional[float]
            The isovalue at which the contour must be plotted. If set to `None` (default)
            a proper isovalue will be set automatically as the 20% of the maximum voxel value.
        positive_color: int
            The color of the positive phase of the plot.
        negative_color: int
            The color of the positive phase of the plot.
        show_negative: bool
            If set to True, will render also the negative part of the Fukui function. (default:
            False)
        filename: Optional[str]
            The name, or the path, of the output `.bmp` file. If `None` (default) the output file
            will be generated from the root of the input filename (e.g. `root.bmp` from `root.xyz`).
        """
        script = self._tcl_script_preamble()

        script += self._tcl_cube_script(
            cubefile,
            isovalue=isovalue,
            positive_color=positive_color,
            negative_color=negative_color,
            show_negative=show_negative,
        )

        root_name = basename(cubefile).removesuffix(".cube")
        filename = filename.removesuffix(".bmp") if filename is not None else root_name
        self._render(script, filename)

    #########################################
    # Interface functions to SPyCCI objects #
    #########################################

    def render_system(self, mol: System, filename: Optional[str] = None) -> None:
        """
        Given a `System` object the function saves a `.bmp` render of its molecular structure.

        Arguments
        ---------
        mol: System
            The `System` object to render.
        filename: Optional[str]
            The name, or the path, of the output `.bmp` file. If `None` (default) the output file
            will be generated from the root of the input filename (e.g. `root.bmp` from `root.xyz`).
        """
        tdir = mkdtemp(prefix=f"{mol.name}_vmd_", dir=os.getcwd())

        with sh.pushd(tdir):

            if filename is None:
                filename = f"{mol.name}_{mol.charge}_{mol.spin}.bmp"

            elif filename.endswith(".dmp"):
                filename += ".bmp"

            mol.geometry.write_xyz(f"{mol.name}.xyz")
            self.render_system_file(f"{mol.name}.xyz", filename="output.bmp")
            shutil.copy("output.bmp", f"../{filename}")
            shutil.rmtree(tdir)

    def render_cube(
        self,
        cube: Cube,
        filename: str,
        isovalue: Optional[float] = None,
        positive_color: int = 1,
        negative_color: int = 0,
        show_negative: bool = False,
    ) -> None:
        """
        Given a `Cube` object the function saves a `.bmp` render of the contained
        volumetric data.

        Arguments
        ---------
        cube: Cube
            The `Cube` object encoding the volumetric data.
        filename: str
            The name, or the path, of the output `.bmp` file (Required because `Cube` objects
            have no pre-assigned names).
        isovalue: Optional[float]
            The isovalue at which the contour must be plotted. If set to `None` (default)
            a proper isovalue will be set automatically as the 20% of the maximum voxel value.
        positive_color: int
            The color of the positive phase of the plot.
        negative_color: int
            The color of the positive phase of the plot.
        show_negative: bool
            If set to True, will render also the negative part of the Fukui function. (default:
            False)
        """
        root_name = filename.removesuffix(".bmp")

        tdir = mkdtemp(prefix=f"{root_name}_vmd_", dir=os.getcwd())

        with sh.pushd(tdir):

            cube.save(f"{root_name}.cube")

            self.render_cube_file(
                f"{root_name}.cube",
                isovalue=isovalue,
                positive_color=positive_color,
                negative_color=negative_color,
                show_negative=show_negative,
                filename="output.bmp",
            )

            shutil.copy("output.bmp", f"../{root_name}.bmp")
            shutil.rmtree(tdir)

    #############################
    # Format specific functions #
    #############################

    def render_fukui_cube(
        self,
        cubefile: str,
        isovalue: Optional[float] = None,
        show_negative: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """
        Given the path to a Fukui function cube file saves a `.bmp` render the volumetric Fukui
        function.

        Arguments
        ---------
        cubefile: str
            The path to the `.fukui.cube` file that must be rendered.
        isovalue: Optional[float]
            The isovalue at which the contour must be plotted. If set to `None` (default)
            a proper isovalue will be set automatically as the 20% of the maximum voxel value.
        show_negative: bool
            If set to True, will render also the negative part of the Fukui function. (default:
            False)
        filename: Optional[str]
            The name, or the path, of the output `.bmp` file. If `None` (default) the output file
            will be generated from the root of the input filename (e.g. `root.bmp` from `root.xyz`).
        """
        root_name = basename(cubefile).removesuffix(".fukui.cube")
        filename = filename.removesuffix(".bmp") if filename is not None else root_name

        self.render_cube_file(
            cubefile,
            isovalue=isovalue,
            positive_color=1,
            negative_color=0,
            show_negative=show_negative,
            filename=filename,
        )

    def render_spin_density_cube(
        self,
        cubefile: str,
        isovalue: Optional[float] = None,
        filename: Optional[str] = None,
    ) -> None:
        """
        Given the path to an ORCA spin density cube file saves a `.bmp` render the function.

        Arguments
        ---------
        cubefile: str
            The path to the `.fukui.cube` file that must be rendered.
        isovalue: Optional[float]
            The isovalue at which the contour must be plotted. If set to `None` (default)
            a proper isovalue will be set automatically as the 20% of the maximum voxel value.
        xyx_rotation: Optional[tuple]
            The tuple of 3 rotation angles (from 0 to 360°) defyning subsequent rotations around
            the X, Y and X axis. If None (default), no rotation is applied.
        filename: Optional[str]
            The name, or the path, of the output `.bmp` file. If `None` (default) the output file will
            be generated from the root of the input filename (e.g. `root_condensed.bmp` from `root.xyz`).
        """
        root_name = basename(cubefile).removesuffix(".spindens.cube")
        filename = filename.removesuffix(".bmp") if filename is not None else root_name

        self.render_cube_file(
            cubefile,
            isovalue=isovalue,
            positive_color=31,
            negative_color=26,
            show_negative=True,
            filename=filename,
        )

    def render_condensed_fukui(
        self,
        cubefile: str,
        filename: Optional[str] = None,
    ) -> None:
        """
        Given the path to a Fukui function cube file saves a `.bmp` render of the condensed Fukui
        functions.

        Arguments
        ---------
        cubefile: str
            The path to the `.fukui.cube` file that must be rendered.
        filename: Optional[str]
            The name, or the path, of the output `.bmp` file. If `None` (default) the output file will
            be generated from the root of the input filename (e.g. `root_condensed.bmp` from `root.xyz`).
        """
        # Check if given file exists
        if isfile(cubefile) is False:
            raise FileNotFoundError(f"Unable to find the {cubefile} file.")

        root_name = basename(cubefile).removesuffix(".fukui.cube")
        script = self._tcl_script_preamble()

        # Load the molecule from the cube file, plot its backbone using Licorice style
        # and color it using the data encoding the condensed Fukui values (saved as partial charges)
        script += f"mol new {cubefile} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}\n"
        script += "mol addrep 0\n"
        script += "mol modstyle 0 0 Licorice 0.1 20.000000 20.000000\n"
        script += "color scale method BWR\n"
        script += "mol modcolor 0 0 Charge\n"
        script += "mol color Charge\n"

        # Add labels near each atom with the condensed Fukui value
        script += "label delete Atoms all\n"
        script += """set all [atomselect 0 "all"]\n"""
        script += "set i 0\n"
        script += "foreach atom [$all get index] {\n"
        script += """    label add Atoms "0/$atom"\n"""
        script += """    label textformat Atoms $i {  (%e) %q}\n"""
        script += """    label textoffset Atoms $i {1.0 0.0}\n"""
        script += """    incr i\n"""
        script += "}\n"
        script += "label textsize 1.\n"
        script += "label textthickness 2\n"
        script += "color Labels Atoms black\n"

        # Apply some final settings
        script += "display cuemode Linear\n"
        script += "mol selection all\n"
        script += "mol material Opaque\n"

        filename = filename.removesuffix(".bmp") if filename is not None else f"{root_name}_condensed"

        self._render(script, filename)

    ####################
    # Internal helpers #
    ####################

    def _render(
        self,
        instructions: str,
        output_name: str,
    ) -> None:
        """
        Given a set of `vmd` instructions, run a render operation outputting a `.bmp`
        image file.

        Arguments
        ---------
        instructions: str
            The string encoding the operations to be executed by `vmd`.
        output_name: str
            The name of the output file (`output_name.bmp`).
        """

        with tmp(mode="w+", suffix=".vmd") as vmd_script:

            vmd_script.write(instructions)
            
            # Apply translation vector 
            vmd_script.write(f"translate by {self.__xyz_translation[0]} {self.__xyz_translation[1]} {self.__xyz_translation[2]}\n")
            
            # Uses an XYX rotation sequence following the proper Euler angle convention. 
            # Rotations are around the camera (screen) axes, which change after each step. 
            # Although the first and third axes are both X, they differ in orientation due 
            # to the intermediate Y rotation, enabling full 3D rotation coverage.
            vmd_script.write(f"rotate x by {self.__xyx_rotation[0]}\n")
            vmd_script.write(f"rotate y by {self.__xyx_rotation[1]}\n")
            vmd_script.write(f"rotate x by {self.__xyx_rotation[2]}\n")
            
            # Apply scale factor
            vmd_script.write(f"scale by {self.__scale}\n")

            if self.shadows:
                vmd_script.write("display shadows on\n")
            if self.ambientocclusion:
                vmd_script.write("display ambientocclusion on\n")
            if self.dof:
                vmd_script.write("display dof on\n")

            vmd_script.write(
                f"""render Tachyon {output_name}.dat "{self.__tachyon_path}" -fullshade -aasamples 12 %s -format BMP -res {self.__resolution[0]} {self.__resolution[1]} -o {output_name}.bmp\n"""
            )
            vmd_script.write("exit\n")

            vmd_script.seek(0)

            if self.suppress_output is True:
                system(f"vmd -dispdev text -e {vmd_script.name}  > /dev/null 2>&1")
            else:
                system(f"vmd -dispdev text -e {vmd_script.name}")

    def _tcl_script_preamble(self) -> str:
        """
        Generates a standard header for the VMD instructions script. The header
        sets orthographic projection, removes axes and sets white background.
        The preamble also sets the carbon color to black.

        Returns
        -------
        str
            The script encoding the script opening
        """
        script = ""
        script += "display projection Orthographic\n"
        script += "display resetview\n"
        script += "color Display Background white\n"
        script += "color Name C black\n"

        if self.show_axes is False:
            script += "axes location Off\n"

        return script

    def _tcl_plot_backbone(self) -> str:
        """
        Generates the code required to plot the molecular backbone of a loaded mol object.

        Returns
        -------
        str
            The script encoding the backbone rendering.
        """
        script = ""
        script += "mol selection all\n"
        script += "mol addrep 0\n"
        script += "mol modstyle 0 0 CPK 1.000000 0.300000 150.000000 12.000000\n"
        script += "mol material Opaque\n"
        script += "mol color Name\n"
        return script

    def _tcl_cube_script(
        self,
        cubefile: str,
        isovalue: Optional[float] = None,
        positive_color: int = 1,
        negative_color: int = 0,
        show_negative: bool = True,
    ) -> str:
        """
        Generates a general script to render a cube file. The user can select the isovalue,
        the color of each phase of the cube file and whether the negative phase is shown.

        Arguments
        ---------
        cubefile: str
            The path to the `.cube` file that must be rendered.
        isovalue: Optional[float]
            The isovalue at which the contour must be plotted. If set to `None` (default)
            a proper isovalue will be set automatically as the 20% of the maximum voxel value.
        positive_color: int
            The color of the positive phase of the plot.
        negative_color: int
            The color of the positive phase of the plot.
        show_negative: bool
            If set to True will plot the negative part of the cube file.

        Raises
        ------
        FileNotFoundError
            Exception raised if the cube file cannot be found.

        Return
        ------
        str
            The script encoding the cube rendering
        """
        # Check if given file exists
        if isfile(cubefile) is False:
            raise FileNotFoundError(f"Unable to find the {cubefile} file.")

        # If not specified by the user automatically compute a guess of the isovalue
        if isovalue is None:
            cube = Cube.from_file(cubefile)
            cmax, cmin = cube.max, cube.min
            isovalue = 0.2 * max(abs(cmax), abs(cmin))

        script = ""

        # Plot the molecular backbone
        script += f"mol new {cubefile} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}\n"
        script += self._tcl_plot_backbone()

        # Print positive part of the isosurface
        script += "mol addrep 0\n"
        script += f"mol modcolor 1 0 ColorID {positive_color}\n"
        script += f"mol modstyle 1 0 Isosurface {isovalue} 0 0 0 1 1\n"
        script += "mol modmaterial 1 0 Translucent\n"
        script += "mol scaleminmax 0 1 0.000000 1.000000\n"

        # Print negative part of the isosurface
        if show_negative is True:
            script += "mol addrep 0\n"
            script += f"mol modcolor 2 0 ColorID {negative_color}\n"
            script += f"mol modstyle 2 0 Isosurface {-isovalue} 0 0 0 1 1\n"
            script += "mol modmaterial 2 0 Translucent\n"
            script += "mol scaleminmax 0 2 -1.000000 0.000000\n"

        # Apply some final settings
        script += "display cuemode Linear\n"

        return script



####################################################################
#               DEFINE VMD-BASED ANIMATION FUNCTIONS               #
####################################################################

def animate(
        systems: Union[List[System], ReactionPath],
        filename: str,
        renderer: Optional[VMDRenderer] = None,
        duration: float = 0.1,
        loop: int = 0,
        reversed: bool = False,
        mirror: bool = False,
        remove_tdir: bool = True,
        suppress_output: bool = False,
) -> None:
    """
    Given a list of `System` objects generate a `.gif` animation by iteratively
    calling `vmd` to render all the frames.

    Arguments
    ---------
    systems: Union[List[System], ReactionPath],
        The ordered list of systems to be rendered in the animation either as a regular
        list object or as a ReactionPath object.
    filename: str
        The name or the path of the output animation `.gif` file.
    renderer: Optional[VMDRenderer]
        The rendered used to generate the animation frames. (default: `VMDRenderer()`)
    duration: float
        The duration in seconds of each frame in the animation. (default: 0.1s)
    loop: int
        The number of time the `.gif` repetes itself. (default: 0 -> Infinite loop)
    reversed: bool
        If set to `True` will reverse the order in which the frames are ordered. The animation will
        start from the last system in the list and progress toward the first. (default: False)
    mirror: bool
        If set to `True`, a reversed copy of the list is appended, with the first and last frames
        removed to avoid duplication at the turning points. The resulting animation is cyclical: it
        progresses from the first frame to the last one, then reverses direction and continues back 
        toward the second frame. (default: False)
    remove_tdir: bool
        If set to `False` will keep the temporary folder containing the render of
        each frame. (default: True)
    suppress_output: bool
        If set to `True` will run the rendering without printing any message from `vmd`. (default: False)
    """
    vmd = renderer if renderer else VMDRenderer()

    if type(vmd) != VMDRenderer:
        raise ValueError(f"The renderer engine must be of type `VMDRenderer`, {type(vmd)} is not a valid renderer")
        
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("PIL").propagate = False

    # Create a temporary directory where the frames will be stored
    tdir = mkdtemp(prefix = "vmd_animation_", dir=os.getcwd())
    
    # Extract the list of system to be used in each step of the aminmation
    steps = systems.systems if isinstance(systems, ReactionPath) else systems

    if reversed is True:
        steps = steps[::-1]

    # Set the output state of the renderer according to user settings
    vmd.suppress_output = suppress_output
    
    # Compute the number of frames per seconds (`fps`) from the user set frame duration
    # Note: This is a workaround since the `duration` keyword is often ignored by `mimsave`
    fps = int(1./duration)

    with sh.pushd(tdir):

        # Render each frame individually using the provided VMD renderer
        frames = []
        for i, system in enumerate(steps):
            vmd.render_system(system, f"frame_{i}.bmp")
            frames.append(imageio.imread(f"frame_{i}.bmp"))
        
        if mirror is True:
            back = deepcopy(frames[::-1])[1:-1]
            frames.extend(back)

        # Join each frame in a single .gif object using the imageio package
        imageio.mimsave("animation.gif", frames, fps=fps, loop=loop)

    # Copy the generated animation to the user-specified location
    shutil.copy(f"{tdir}/animation.gif", filename)

    # If required by the user remove the temporary directory
    if remove_tdir:
        shutil.rmtree(tdir)

        
def animate_normal_mode(
    system: System,
    mode: int,
    filename: str,
    displacement: float = 0.5,
    steps: int = 10,
    renderer: Optional[VMDRenderer] = None,
    duration: float = 0.05,
    remove_tdir: bool = True,
    suppress_output: bool = False,
) -> None:
    """
    Animate the normal mode of vibration of a molecular system. The function automatically
    generates a series of `System` objects corresponding to the molecular geometries displaced
    along a chosen normal mode and produces a `.gif` animation of the motion.

    Parameters
    ----------
    system : System
        The molecular system whose normal mode is to be animated. (Must contain vibrational data.)
    mode : int
        Index of the normal mode to animate.
    filename : str
        The output filename, including path, for the generated `.gif` animation.
    displacement : float, optional
        Maximum amplitude of displacement along the normal mode (default: 0.5 Å).
    steps : int, optional
        Number of intermediate frames to generate along the displacement path (default: 10).
    renderer: Optional[VMDRenderer]
        The rendered used to generate the animation frames. (default: `VMDRenderer()`)
    duration: float
        The duration in seconds of each frame in the animation. (default: 0.1s)
    remove_tdir: bool
        If set to `False` will keep the temporary folder containing the render of
        each frame. (default: True)
    suppress_output : bool, optional
        If `True`, suppress messages from the VMD renderer during frame generation (default: False).
    """
    # Check if the given system is of type `System`
    if isinstance(system, System) is False:
        raise TypeError(f"The system argument must be of type `System`, type `{type(system)}` given instead.")

    # Check if vibrational data are available
    data: VibrationalData = system.properties.vibrational_data
    if data is None:
        raise RuntimeError(f"Vibrational data not found for the `{system.name}` system.")

    # Check if the provided mode index is valid and extract the corresponding normal mode
    nmodes = len(data.frequencies)
    if mode < 0 or mode >= nmodes:
        raise IndexError(f"Normal mode {mode} index out of bounds [0, {nmodes}].")

    normal_mode: np.ndarray = data.normal_modes[mode]

    # Check if the mode corresponds to zero frequency, if yes warn the user
    if np.isclose(data.frequencies[mode], 0.0, atol=1e-12):
        logger.warning(f"The selected mode {mode} is associated with a zero frequency.")

    syslist: List[System] = []
    for l in np.linspace(-displacement, displacement, steps):

        # Copy the molecular geometry data
        geom = deepcopy(system.geometry)

        # Displace the system geometry along the selected normal mode
        coords: np.ndarray = np.array(geom.coordinates)
        coords += l * normal_mode.reshape(geom.atomcount, 3)
        geom.set_coordinates(coords)

        # Create a new system object and append it to the systems list
        frame = System(f"d={l:.2f}", geom, charge=system.charge, spin=system.spin)
        syslist.append(frame)

    # Call the animation tool
    animate(
        syslist,
        filename,
        renderer=renderer,
        duration=duration,
        mirror=True,
        remove_tdir=remove_tdir,
        suppress_output=suppress_output,
    )
