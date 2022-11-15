/////////////////////////////////////////////////////////////////
///////////////  Constants   ////////////////////////////////////
/////////////////////////////////////////////////////////////////


// TODO:
// afficher pi et rho (a la plac de gmma, avecc un switch)

data_name = 'all_30_min_interval_july'; //all_10_min_interval_july_1515_1615, all_10_min_interval_july_404_504

// all_15_min_interval, center_15_min_interval, all_30_min_interval_july

data_file = './alluvial_data/' + data_name +  '_data.json';
image_path = './images/' + data_name + '/';

onlyTickHour = false;
delta_ticks_fs = 3.;

// initial values for important params for svg
nodeWidth = 20;
nodePadding = 10;
widthPerTimestep = 150;
heightPerCluster = 80;
ticksFontSize = 10;

// old
//nodeWidth = 60;
//nodePadding = 20;
//widthPerTimestep = 220;
//heightPerCluster = 330;
//ticksFontSize = 45;


initMode = 'entry';

xMargin = 250;
yTopMargin = 100;
yBottomMargin = 25;

margins = {
  'left': xMargin,
  'right': xMargin,
  'top': yTopMargin,
  'bottom': yBottomMargin
};

nodeRx = 4;
nodeRy = 4;

nIterationsSankey = 50;

// Can be replaced by 'none', 'path', 'input' or 'output'
edgeColor = 'input'

// the number associated to the emtpy cluster
// set to -1 to show the empty cluster
emptyCluster = -1;

minFlowStation = 2;
minFlowBlock = 50;

offsetNodeName = 10;
xOffsetTooltip = - 200;
yOffsetTooltip = 50;

linkStrokeOpacity = .5;
linkOpacity = .5;
nodeOpacity = .8;
nodesOpaqueOpacity = .15;

timestepScrollCorrectFactor = .995;


/////////////////////////////////////////////////////////////////
///////////////  Functions   ////////////////////////////////////
/////////////////////////////////////////////////////////////////

function hideShowSankeySettings(button){
  if (button.value === 'hidden'){
    document.getElementById('selectSankeyOptions').style.display = 'inline';
    button.innerHTML = 'Hide Sankey Settings';
    button.value = 'shown';
  }
  else if (button.value == 'shown'){
    document.getElementById('selectSankeyOptions').style.display = 'none';
    button.innerHTML = 'Show Sankey Settings';
    button.value = 'hidden';
  }
}

// Just a UUID generator
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

var processLineText = function(o) {
  frac = o[0]
  stations = o[1]
  codes = o[2]

  content = codes.map((v, i) => v + ' - ' + stations[i])
  
  content = '<br>' + JSON.stringify(content)
  .replace(/"/g, '')
  .replace(/,/g, '</br><br>')
  .replace('[', '')
  .replace(']', '')
  
  return {
    title:frac,
    content: content
  };
}

function tooltipContentNode(d, mode){

  units = (mode === 'block')? 'OD pairs': 'stations'

  s = '<b>Cluster</b>: ' + d.clusterName + '<br>' +
  '<b>Date</b>: ' + new Date(d.datetime).toLocaleString() + '<br>' +
  '<b>Number of ' + units + '</b>: ' + d.value + '<br>' +
  '<b>Arrivals</b>: ' + d.arrivals + '<br>' +
  '<b>Departures</b>: ' + d.departures + '<br>' +
  '<b>From inactive cluster</b>: ' + d.flowFromEmptyCluster
  return s;
};

function tooltipContentLink(d, mode){
  units = (mode === 'block')? 'OD pairs': 'stations'

  s = '<b>Source</b>: ' + d.source.clusterName +'<br>' +
  '<b>Target</b>: ' + d.target.clusterName + '<br>' +
  '<b>Number of ' + units + '</b>: ' + d.value + '<br>'  +
  '<b>Date</b>: ' + new Date(d.target.datetime).toLocaleString() 
  return s;
};

function initSankeyOptionRanges(nodeWidth, nodePadding, widthPerTimestep, heightPerCluster){
  document.getElementById('nodeWidthText').innerHTML = 'node width = ' + nodeWidth;
  document.getElementById('nodePaddingText').innerHTML = 'node padding = ' + nodePadding;
  document.getElementById('widthText').innerHTML = 'width = ' + widthPerTimestep;
  document.getElementById('heightText').innerHTML = 'height = ' + heightPerCluster;

  document.getElementById('sankeyNodeWidth').value = nodeWidth;
  document.getElementById('sankeyPadding').value = nodePadding;
  document.getElementById('sankeyWidth').value = widthPerTimestep;
  document.getElementById('sankeyHeight').value = heightPerCluster;
}

function selectMode(val){

  if (val === 'exit'){
    document
    .getElementById('title')
    .innerHTML = '<b>Clusters of exit stations over time</b>';

    document.getElementById('entryRoutemapContainer')
    .appendChild(document.getElementById('details'))
  }
  else if (val == 'entry'){
    document
    .getElementById('title')
    .innerHTML = '<b>Clusters of entry stations over time</b>';

    document.getElementById('exitRoutemapContainer')
    .appendChild(document.getElementById('details'))
  }
  else if (val == 'block'){
    document
    .getElementById('title')
    .innerHTML = '<b>Blocks over time</b>';
  };

  updateSankeyMode(val);

  document.getElementById('details')
  .style.cssText = 'display: none';

}

function updateSankeyMode(val) {  
  nodeWidth = document.getElementById('sankeyNodeWidth').value;
  nodePadding = document.getElementById('sankeyPadding').value;
  widthPerTimestep = document.getElementById('sankeyWidth').value;
  heightPerCluster = document.getElementById('sankeyHeight').value;

  d3.json(data_file)
  .then(d => prepareHtml(d, val));

  d3.json(data_file)
  .then(drawSVGAndAnimate(val, nodeWidth, nodePadding, widthPerTimestep, heightPerCluster));
}

function updateSankeyNodeWidth(val) {
  document.getElementById('nodeWidthText').innerHTML = 'node width = ' + val;

  mode = document.getElementById('selectMode').value;
  nodePadding = document.getElementById('sankeyPadding').value;
  widthPerTimestep = document.getElementById('sankeyWidth').value;
  heightPerCluster = document.getElementById('sankeyHeight').value;

  d3.json(data_file)
  .then(drawSVGAndAnimate(mode, val, nodePadding, widthPerTimestep, heightPerCluster));
}

function updateSankeyNodePadding(val) {
  document.getElementById('nodePaddingText').innerHTML = 'node padding = ' + val;

  mode = document.getElementById('selectMode').value;
  nodeWidth = document.getElementById('sankeyNodeWidth').value;
  widthPerTimestep = document.getElementById('sankeyWidth').value;
  heightPerCluster = document.getElementById('sankeyHeight').value;

  d3.json(data_file)
  .then(drawSVGAndAnimate(mode, nodeWidth, val, widthPerTimestep, heightPerCluster));
}

function updateSankeyWidth(val) {
  document.getElementById('widthText').innerHTML = 'width = ' + val;

  mode = document.getElementById('selectMode').value;
  nodeWidth = document.getElementById('sankeyNodeWidth').value;
  nodePadding = document.getElementById('sankeyPadding').value;
  heightPerCluster = document.getElementById('sankeyHeight').value;

  d3.json(data_file)
  .then(drawSVGAndAnimate(mode, nodeWidth, nodePadding, val, heightPerCluster));
}

function updateSankeyHeight(val) {
  document.getElementById('heightText').innerHTML = 'height = ' + val;

  mode = document.getElementById('selectMode').value;
  nodeWidth = document.getElementById('sankeyNodeWidth').value;
  nodePadding = document.getElementById('sankeyPadding').value;
  widthPerTimestep = document.getElementById('sankeyWidth').value;

  d3.json(data_file)
  .then(drawSVGAndAnimate(mode, nodeWidth, nodePadding, widthPerTimestep, val));
}

function drawSVGAndAnimate(mode, nodeWidth, nodePadding, widthPerTimestep, heightPerCluster){
  return d => drawSVGAndAnimatep(d, mode, nodeWidth, nodePadding, widthPerTimestep, heightPerCluster);
}

function updateTickFs(ticksFontSize){
  document
  .getElementById('ticksFontSizeText')
  .innerHTML = 'tick font size = ' + ticksFontSize;

  document
  .getElementById('sankeyTickFs')
  .value = ticksFontSize;

  d3.selectAll('.tick')
  .style('font-size', ticksFontSize + 'px');
};

function prepareHtml(data, mode) {

  // data-dependant html structure

  // select station or a group of stations
  // and see its trajectory
  d3.select('#selectStationOptionsStat')
  .selectAll('select')
  .remove();

  d3.select('#selectStationOptionsStat')
  .append('select')
  .attr('class','select')
  .attr('id', 'selectStation')

  d3.select('#selectStationOptionsLine')
  .selectAll('select')
  .remove();

  d3.select('#selectStationOptionsLine')
  .append('select')
  .attr('class','select')
  .attr('id', 'selectLine')

  d3.select('#selectCluster')
  .selectAll('select')
  .remove();

  d3.select('#selectCluster')
  .append('select')
  .attr('class','select')
  .attr('id', 'selectClusterVal')

  // cluster selection options
  d3.select('#selectClusterVal')
  .selectAll('option')
  .remove()

  if (mode === 'block'){
    document.getElementById('selectStationOptionsStat')
    .style.cssText = 'display: none';
    document.getElementById('selectStationOptionsLine')
    .style.cssText = 'display: none';
    document.getElementById('selectCluster')
    .style.cssText = 'display: none';
  }
  else{
    document.getElementById('selectStationOptionsStat')
    .style.cssText = 'display: inherit';
    document.getElementById('selectStationOptionsLine')
    .style.cssText = 'display: inherit';
    document.getElementById('selectCluster')
    .style.cssText = 'display: inherit';

    stationNames = data['stations'].sort(function (s1, s2){
      return s1.toLowerCase() > s2.toLowerCase() ? 1: -1
    });
    stationNames = ['None'].concat(stationNames);
    lines = data['lines'];

    if (mode =='entry'){
      nbClusters = data['nbEntryClusters']
    } 
    else if (mode === 'exit'){
      nbClusters = data['nbExitClusters']
    }
    clusterList = ['None'].concat([...Array(nbClusters).keys()])
    
    // stations selection options
    d3.select('#selectStation')
    .selectAll('option')
    .data(stationNames).enter()
    .append('option')
    .text(d => d);

    // lines selection options
    d3.select('#selectLine')
    .selectAll('option')
    .data(['None'].concat(Object.keys(lines))).enter()
    .append('option')
    .text(d => d);

    d3.select('#selectClusterVal')
    .selectAll('option')
    .data(clusterList).enter()
    .append('option')
    .text(d => d);

    d3.select('#details')
    .selectAll('div').remove();

    // station details div
    for (line in lines){

      var lineDiv = d3.select('#details')
      .append('div')
      .attr('class', 'lineContainer')
      .attr('id', 'lineContainer' + line)

      lineDiv
      .append('div')
      .attr('class', 'lineTitleContainer')
      .attr('id', 'lineTitleContainer' + line)
      .text(line)

      lineDiv 
      .append('div')
      .attr('class', 'lineContentContainer')
      .attr('id', 'lineContentContainer' + line)
    }
  };
}

// Define Sankey function that will return
// the nodes and links that will be 
// used in the SVG
var sankey = function(
  nodeWidth,
  nodePadding,
  margins){
  return d3.sankey()
  .nodeId(d => d.name)
  .nodeAlign(d3.sankeyCenter)
  .nodeWidth(nodeWidth)
  .nodePadding(nodePadding)
  .nodeSort((n1, n2) => n1.cluster >= n2.cluster? 1: -1)
  .iterations(nIterationsSankey)
  .extent(
    [[margins.left, margins.bottom],
     [width - margins.right, height - margins.top]])
}


function drawSVGAndAnimatep(
  data,
  mode,
  nodeWidth,
  nodePadding,
  widthPerTimestep,
  heightPerCluster) {

  console.log(data);

  d3.select('#sankeySVG').remove();
  d3.select('#tooltip').remove();

  nbTimesteps = data['nbTimesteps'];
  timesteps = data['timesteps'];
  daysOfWeek  = data['daysOfWeek'];
  stationNames = data['stations'];
  stationNames = ['None'].concat(stationNames);
  lines = data['lines'];

  if (mode === 'entry'){
    nbClusters = data['nbEntryClusters'];
    trajectories = data['entryTrajectories'];
    cmap = data['entryClusterCmap'];
    height = nbClusters * heightPerCluster;
  } else if (mode === 'exit'){
    nbClusters = data['nbExitClusters'];
    trajectories = data['exitTrajectories'];
    cmap = data['exitClusterCmap'];
    height = nbClusters * heightPerCluster;
  } else if (mode === 'block'){
    nbClusters = data['nbBlocks'];
    trajectories = null;
    cmap = data['blockCmap'];
    height = .3 * nbClusters * heightPerCluster;
  };

  function color(d){
    return cmap[d.cluster];
  }
  width = nbTimesteps * widthPerTimestep;

  // Initialize SVG
  svg = d3.select('#sankeyDiagram')
  .append('svg')
  .attr('id', 'sankeySVG')
  .attr('viewBox', [0, 0, width, height])
  .attr('class', 'svg-background');

  // create a tooltip
  var tooltip = d3.select('#sankeyDiagram')
  .append('div')
  .attr('id', 'tooltip')
  .style('opacity', 0)
  .attr('class', 'tooltip')

  // Add zoom
  function svgZoom(){
    svg.attr('transform', d3.event.transform)
  }
  var zoom = d3.zoom()
  .scaleExtent([.1, 1.5])
  .on('zoom', svgZoom);
  d3.select('#sankeyDiagram').call(zoom);

  // Retrieve D3-generated nodes and links (i.e. positioned in the page)
  // nodes and links are js object returned by d3-sankey
  // that will be used to create the SVG elements
  // node and link
  var sankeyF = sankey(nodeWidth, nodePadding, margins);

  if (mode === 'entry'){
    var {nodes, links} = sankeyF({
      nodes: data.entryNodes.map(d => Object.assign({}, d)),
      links: data.entryLinks.map(d => Object.assign({}, d))
    });
  } else if (mode === 'exit'){
    var {nodes, links} = sankeyF({
      nodes: data.exitNodes.map(d => Object.assign({}, d)),
      links: data.exitLinks.map(d => Object.assign({}, d))
    });
  } else if (mode === 'block'){
    var {nodes, links} = sankeyF({
      nodes: data.blockNodes.map(d => Object.assign({}, d)),
      links: data.blockLinks.map(d => Object.assign({}, d))
    });
  }

  minFlow = (mode === 'block') ? minFlowBlock : minFlowStation;

  console.log(nodes);
  console.log(links);

  // map the node data to SVG
  // creates a <g> for each node (=cluster)
  // that will contain two rects : 
  // a regular one and an opaque one
  // the opaque one will be used to represent
  // trajectories for a group of stations
  const nodeGroup = svg.append('g')
  .selectAll('g')
  .data(nodes)  // Data is linked here with DOM element
  .join(
    enter => enter
    .filter(d => d.cluster != emptyCluster)
    .append('g')
  );

  const nodeOpaque = nodeGroup
  .append('rect')
  .attr('x', d => d.x0)
  .attr('y', d => d.y0)
  .attr('index', d => d.index)
  .attr('rx', nodeRx)
  .attr('ry', nodeRy)
  .attr('height', d => 0.)
  .attr('width', d => d.x1 - d.x0)
  .attr('class', 'nodesOpaque')
  .attr('fill', d => color(d))
  .style('opacity', nodesOpaqueOpacity)
  .attr('pointer-events', 'none');

  const node = nodeGroup
  .append('rect')
  .attr('x', d => d.x0)
  .attr('y', d => d.y0)
  .attr('index', d => d.index)
  .attr('rx', nodeRx)
  .attr('ry', nodeRy)
  .attr('height', d => d.y1 - d.y0)
  .attr('width', d => d.x1 - d.x0)
  .attr('class', 'nodes')
  .attr('fill', d => color(d))
  .style('opacity', nodeOpacity);

  // Add link elements to SVG
  const link = svg.append('g')
  .attr('fill', 'none')
  .attr('opacity', linkOpacity)
  .attr('stroke-opacity', linkStrokeOpacity)
  .selectAll('g')
  .data(links)
  .join('g')
  .filter(d => d.value >= minFlow)
  .filter(d => d.source.cluster != emptyCluster)
  .filter(d => d.target.cluster != emptyCluster)
  .style('mix-blend-mode', 'multiply');

  // Define the correct color or gradient to the link
  if (edgeColor === 'path') {
    const gradient = link.append('linearGradient')
    .attr('id', d => (d.uid = uuidv4()))
    .attr('gradientUnits', 'userSpaceOnUse')
    .attr('x1', d => d.source.x1)
    .attr('x2', d => d.target.x0);

    gradient.append('stop')
    .attr('offset', '0%')
    .attr('stop-color', d => color(d.source));

    gradient.append('stop')
    .attr('offset', '100%')
    .attr('stop-color', d => color(d.target));
  }

  // Apply the actual path in the link
  link.append('path')
  .attr('d', d3.sankeyLinkHorizontal())
  .attr('stroke', d => edgeColor === 'none' ? '#aaa'
    : edgeColor === 'path' ? `url(#${d.uid}`
    : edgeColor === 'input' ? color(d.source)
    : color(d.target))
  .attr('stroke-width', d => Math.max(1, d.width));


  // highlights the trajectory of a station
  d3.select('#selectStation')
  .on('change',onchangeStation)
  d3.select('#selectLine')
  .on('change',onchangeLine)

  function onchangeStation() {
    selectValue = d3.select('#selectStation').property('value');
    svg.selectAll('rect')
    .style('stroke', 'none');
    
    if (selectValue !== 'None'){
      selectTraj = trajectories[selectValue];
      svg.selectAll('rect')
      .filter(d => selectTraj.includes(d.name))
      .style('stroke', 'black');
    }
  };

  function onchangeLine() {
    selectValue = d3.select('#selectLine').property('value');

    node
    .attr('height', 0.) 

    nodeOpaque
    .attr('y', d => d.y0)
    .attr('height', d => d.y1 - d.y0) //d.y1 - d.y0

    if (selectValue !== 'None'){
      selectStations = lines[selectValue]
      selectTraj = new Set;
      for (let i = 0; i < selectStations.length; i++){
        stat = selectStations[i]
        traj = new Set(trajectories[stat])
        selectTraj = new Set([...selectTraj, ...traj]);
      }
      selectTraj = Array.from(selectTraj);

      node
      .filter(d => selectTraj.includes(d.name))
      .attr('y', d => d.y0)
      .attr('height', d => d.lineRatio[selectValue] * (d.y1 - d.y0));
      //.style('opacity', nodeOpacity)

      nodeOpaque
      .filter(d => selectTraj.includes(d.name))
      .attr('y', d => d.y0 + d.lineRatio[selectValue] * (d.y1 - d.y0))
      .attr('height', d => (1 - d.lineRatio[selectValue]) * (d.y1 - d.y0))
    }
    else {
      nodeOpaque
      .attr('y', d => d.y0)
      .attr('height', 0.) 

      node
      .attr('y', d => d.y0)
      .attr('height', d => d.y1 - d.y0)
      .style('opacity', nodeOpacity)
    }
  };

  // 4 functions that change the tooltip when user hover / move / leave a cell
  var nodeMouseover = function(d) {

    tooltip.style('opacity', 1);

    // Fires when the mouse enters the canvas or any of its children.
    if (d3.select('#selectLine').property('value') === 'None' &
        d3.select('#selectStation').property('value') === 'None'){  

      d3.select(this)
      .style('stroke', 'black')
      .style('opacity', 1.);

      mode = document.getElementById('selectMode').value;

      if (mode !== 'block'){
        document.getElementById(
        ( mode === 'entry'? 'exit': 'entry') + 'RoutemapImage'
        ).style.cssText = 'display: none';
        document.getElementById('details')
        .style.cssText = 'display: inherit';
      };
    }
  }

  var linkMouseover = function(d) {
    // Fires when the mouse enters the canvas or any of its children.
    tooltip.style('opacity', 1);

    mode = document.getElementById('selectMode').value;

    if (mode !== 'block'){
      document.getElementById(
        (mode === 'entry'? 'exit': 'entry') + 'RoutemapImage'
      ).style.cssText = 'display: none';
      document.getElementById('details')
      .style.cssText = 'display: inherit';
    };
  }

  var linkMousemove = function(d) {
    //  Fires on any mouse movement over the canvas
    tooltip.html(tooltipContentLink(d, mode))
    .style('position', 'absolute')
    .style('left', (d3.event.pageX + xOffsetTooltip) + 'px')
    .style('top', (d3.event.pageY + yOffsetTooltip) + 'px');

    if (mode !== 'block'){
      for (line in lines){
        const {title, content} = processLineText(d.stationsLink[line]);
        d3.select('#lineTitleContainer' + line)
        .html(line + ' : ' + title);
        d3.select('#lineContentContainer' + line)
        .html(content);
      }
    };
  }

  var nodeMousemove = function(d) {
    //  Fires on any mouse movement over the canvas
    tooltip.html(tooltipContentNode(d, mode))
    .style('position', 'absolute')
    .style('left', (d3.event.pageX + xOffsetTooltip) + 'px')
    .style('top', (d3.event.pageY + yOffsetTooltip) + 'px');

    if (mode !== 'block'){
      for (line in lines){
        const {title, content} = processLineText(d.stationsNode[line])
        d3.select('#lineTitleContainer' + line)
        .html(line + ' : ' + title);
        d3.select('#lineContentContainer' + line)
        .html(content);
      }
    };
  }

  var nodeMouseleave = function(d) {
    if (d3.select('#selectLine').property('value') === 'None' &
        d3.select('#selectStation').property('value') === 'None'){
      
      tooltip.style('opacity', 0);
      
      d3.select(this)
      .style('stroke', 'none')
      .style('opacity', nodeOpacity);

      document.getElementById(
        (mode === 'entry'? 'exit': 'entry') + 'RoutemapImage'
      ).style.cssText = 'display: inherit';
      document.getElementById('details')
      .style.cssText = 'display: none';
    }
  }

  var linkMouseleave = function(d) {
    tooltip.style('opacity', 0);

    document.getElementById(
      (mode === 'entry'? 'exit': 'entry') + 'RoutemapImage'
    ).style.cssText = 'display: inherit';
    document.getElementById('details')
    .style.cssText = 'display: none';
  }

  var click = function(d){
    document.getElementById('routemapTimestepRange').value = d.timestep
    updateRoutemap();
  }

  link
  .on('mouseover', linkMouseover)
  .on('mousemove', linkMousemove)
  .on('mouseleave', linkMouseleave)
  .on('click', click);

  node
  .on('mouseover', nodeMouseover)
  .on('mousemove', nodeMousemove)
  .on('mouseleave', nodeMouseleave)
  .on('click', click);

  // Create scale
  //var x = d3.scaleTime()
  //.rangeRound([margins.left, width - margins.right])
  //.domain([new Date(timesteps[0]), new Date(timesteps[timesteps.length - 1])])

  var ordinalRange = [];
  var tickVals = [];

  start = parseFloat(margins.left) + .5 * nodeWidth;

  // hacky way to find appropriate step
  step = 0.;
  x0 = node._groups[0][0].x.animVal.value;
  for (const rect of node._groups[0]) {
    val = rect.x.animVal.value;
    if (val > x0){
      step = ((val - x0) * 12 * 100) / timesteps.length; // why 100??   * 100 for 1 day, *12 * 100 for 5 weeks
      break;
    }
  }

  step = step + .14735 * nodeWidth; // minor correction

  for (i = 0; i < timesteps.length; i++){
    
    ordinalRange.push(start + i * step)

    arrT = timesteps[i].split('T');
    dateT = arrT[0];
    timeT = arrT[1];
    if (onlyTickHour){
      tickVals.push(timeT);
    } else {
      tickVals.push(dateT.slice(5) + '-' + timeT);
    }
  }

  var x = d3.scaleOrdinal()
  .range(ordinalRange)
  .domain(timesteps);
  
  // Add scales to axis
  var xAxis = d3.axisBottom(x)
  .tickValues(tickVals)
  //.ticks(Math.floor(nbTimesteps / 2))
  .tickSize(25)
  //.tickFormat(d3.timeFormat('%H:%M'));

  // Append group and insert axis
  svg.append('g')
  .attr('id', 'ticks')
  .attr('class', 'axis')
  .attr('transform', 'translate(0,' + (height - margins.top * .8) + ')')
  .call(xAxis)
  .selectAll("text")  
    .style("text-anchor", "end")
    .attr("dx", "-.8em")
    .attr("dy", ".15em")
    .attr("transform", "rotate(-65)");

  //ticksFontSize = document.getElementById('sankeyTickFs').value;
  updateTickFs(ticksFontSize);

  mode = d3.select('#selectMode').property('value');
  document
  .getElementById('selectClusterTxt')
  .innerHTML = '<em>Select ' + mode + ' cluster&nbsp&nbsp</em>';


  // Update the routemap picture according
  // to the slider
  // Note that the slider is in [1: T]
  // but prints timesteps
  d3.select('#routemapTimestepRange')
  .property('max', timesteps.length - 1)
  .on('input', updateRoutemap);

  d3.select('#selectCluster')
  .on('input', updateRoutemap);

  d3.select('#showFullGammaButton')
  .on('click', updateFullGamma);

  function updateFullGamma(){
    button = document
    .getElementById('showFullGammaButton');

    if (button.value === 'hide'){
      button.innerHTML = 'Show full cluster interactions';
      button.value = 'show';
    }
    else if (button.value == 'show'){
      button.innerHTML = 'Show current cluster interactions';
      button.value = 'hide';
    }
    updateRoutemap();
  }

  function updateRoutemap(){
    t = d3.select('#routemapTimestepRange').property('value');
    k = d3.select('#selectClusterVal').property('value');
    mode = d3.select('#selectMode').property('value');

    document
    .getElementById('selectClusterTxt')
    .innerHTML = '<em>Select ' + mode + ' cluster&nbsp&nbsp</em>';


    fullGamma = (document
    .getElementById('showFullGammaButton')
    .value == 'show'? false : true);

    datetime = timesteps[t].split('T');
    date = datetime[0];
    time = datetime[1];
    dayOfWeek = daysOfWeek[t];
    tickValue = tickVals[t];

    document
    .getElementById('routemapTimestepRangeText')
    .innerHTML = dayOfWeek + ' ' + date + ' ' + time;

    ticks = document.getElementsByClassName('tick');
    fs = document.getElementById('sankeyTickFs').value;
    selected_fs = delta_ticks_fs + parseFloat(fs);

    for (i = 0; i < ticks.length; i++){
      ticks[i].style.fontWeight = 'normal';
      ticks[i].style.fontSize = fs + 'px';

      if (ticks[i].getElementsByTagName('text')[0].innerHTML == tickValue){
        ticks[i].style.fontWeight = 'bold';
        ticks[i].style.fontSize = selected_fs + 'px';
      }
    }
  
    if (mode === 'entry'){
      document
      .getElementById('entryRoutemapImage')
      .setAttribute('src', image_path + 'routemap_entries_' + t + '_' + k + '.jpg');

      if (k === 'None'){
        document
        .getElementById('exitRoutemapImage')
        .setAttribute('src', image_path + 'routemap_exits_' + t + '_None.jpg');
      } else {
        document
        .getElementById('exitRoutemapImage')
        .setAttribute('src', image_path + 'routemap_exits_gamma_' + t + '_' + k + '.jpg');
      }
    } else if (mode === 'exit'){
      if (k === 'None'){
        document
        .getElementById('entryRoutemapImage')
        .setAttribute('src', image_path + 'routemap_entries_' + t + '_None.jpg');
      } else {
        document
        .getElementById('entryRoutemapImage')
        .setAttribute('src', image_path + 'routemap_entries_gamma_' + t + '_' + k + '.jpg');
      }
      document
      .getElementById('exitRoutemapImage')
      .setAttribute('src', image_path + 'routemap_exits_' + t + '_' + k + '.jpg');

    } else if (mode === 'block'){
      document
      .getElementById('entryRoutemapImage')
      .setAttribute('src', image_path + 'routemap_entries_' + t + '_None.jpg');
      document
      .getElementById('exitRoutemapImage')
      .setAttribute('src', image_path + 'routemap_exits_' + t + '_None.jpg');
    };

    //document
    //.getElementById('gammaImage')
    //.setAttribute('src', image_path + 'gamma_' + mode + '_' + k + '.jpg');

    if (fullGamma) {
      document
      .getElementById('gammaImage')
      .setAttribute('src', image_path + 'gamma_full.jpg');
    } else {
      if ((mode === 'entry') || (mode === 'exit')){ 
        document
        .getElementById('gammaImage')
        .setAttribute('src', image_path + 'gamma_interact_'+ t + '_' + mode + '_' + k + '.jpg');
      } else if (mode === 'block'){
        document
        .getElementById('gammaImage')
        .setAttribute('src', image_path + 'gamma_interact_'+ t + '_exit_None.jpg');
      };
    };

    // translate SVG with timestep cursor
    svgbb = document.getElementById('sankeySVG').getBoundingClientRect();
      
    svgWidth = timestepScrollCorrectFactor * parseFloat(svgbb.width);
    xTranslateVal = parseFloat(margins.left) - (t / timesteps.length) * svgWidth;
    svg.attr('transform', 'translate(' + xTranslateVal + ',0)');
  }
  // initial call
  updateRoutemap();
}

// waiting for data

// non D3js html writing
initSankeyOptionRanges(nodeWidth, nodePadding, widthPerTimestep, heightPerCluster);

d3.json(data_file)
.then(d => prepareHtml(d, initMode));

// first sankey construction
// can be later overwritten by interactively
// changing the sankey options
//d3.json(data_file)
//.then(drawSVGAndAnimate(mode, nodeWidth, nodePadding, widthPerTimestep, heightPerCluster));


selectMode(initMode);


